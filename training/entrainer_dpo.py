"""Je poursuis le LoRA SFT avec un pilote DPO TRL, référencé au SFT figé."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'

import mlflow
import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from training.entrainer_lora import JournalMLflow, empreinte, lire_jsonl
from training.format_sft import FORMAT_SFT


def empreinte_adaptateur(modele, nom):
    """Vérifie les poids d'un adaptateur, indépendamment de son nom interne."""
    h = hashlib.sha256()
    for cle, poids in sorted(get_peft_model_state_dict(modele, adapter_name=nom).items()):
        h.update(cle.encode())
        h.update(poids.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def verifier_tokenisation(dataset, source, tokenizer, longueur_max):
    """Contrôle ce que TRL utilise réellement : prompt, réponse et EOS complets."""
    assert len(dataset) == len(source), 'TRL a supprimé des exemples.'
    for ligne, origine in zip(dataset, source):
        prompt = tokenizer.encode(origine['prompt'], add_special_tokens=False)
        assert ligne['prompt_ids'] == prompt, origine['id']
        for cle in ('chosen', 'rejected'):
            attendus = tokenizer.encode(origine['prompt'] + origine[cle], add_special_tokens=False)
            assert prompt + ligne[f'{cle}_ids'] == attendus, origine['id']
            assert attendus[-1] == tokenizer.eos_token_id and len(attendus) <= longueur_max


def main():
    """Trace les données, contrôle la référence puis entraîne 32 pas de DPO."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracking-uri', required=True)
    parser.add_argument('--artifact-location', required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Le dossier DPO contient déjà un run ; je refuse de l’écraser.')
    args.output.mkdir(parents=True, exist_ok=True)
    origine = json.loads((args.adapter.parent / 'protocole_entrainement.json').read_text())
    assert origine['format_sft'] == FORMAT_SFT and origine['arguments']['revision'] == args.revision
    manifeste = json.loads((args.data_dir / 'manifest.json').read_text())
    assert manifeste['revision'] == args.revision and manifeste['max_length'] == 4096
    donnees = {s: lire_jsonl(args.data_dir / f'{s}.jsonl') for s in ('train', 'validation')}
    for split in donnees:
        assert empreinte(args.data_dir / f'{split}.jsonl') == manifeste['splits'][split]['sha256']
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base', revision=args.revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    socle = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B-Base', revision=args.revision, dtype=torch.bfloat16)
    socle.config.use_cache = False
    modele = PeftModel.from_pretrained(socle, str(args.adapter), is_trainable=True)
    empreinte_initiale = empreinte_adaptateur(modele, 'default')
    config = DPOConfig(
        output_dir=str(args.output / 'checkpoints'), max_steps=32,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=5e-6, beta=0.1,
        loss_type=['sigmoid'], lr_scheduler_type='cosine', warmup_steps=2,
        max_length=4096, bf16=True, tf32=True, gradient_checkpointing=True,
        precompute_ref_log_probs=True, precompute_ref_batch_size=2,
        eval_strategy='steps', eval_steps=16, save_strategy='steps', save_steps=16,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, logging_steps=4, logging_first_step=True,
        report_to='none', seed=3407, data_seed=3407, optim='adamw_torch_fused',
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    nom_experience = 'OCR14-Colab-DPO-v3'
    if client.get_experiment_by_name(nom_experience) is None:
        client.create_experiment(nom_experience, artifact_location=args.artifact_location)
    mlflow.set_experiment(nom_experience)
    with mlflow.start_run(run_name='dpo-pilote-256-paires-32-pas') as run:
        debut = time.perf_counter()
        trainer = DPOTrainer(model=modele, args=config, processing_class=tokenizer,
            train_dataset=Dataset.from_list(donnees['train']),
            eval_dataset=Dataset.from_list(donnees['validation']), callbacks=[JournalMLflow()])
        # TRL 1.17.1 copie le LoRA initial dans « ref ». Le socle seul serait une mauvaise référence.
        assert 'ref' in modele.peft_config
        assert empreinte_adaptateur(modele, 'ref') == empreinte_initiale
        assert not any(p.requires_grad for n, p in modele.named_parameters() if '.ref.' in n)
        for split, ds in (('train', trainer.train_dataset), ('validation', trainer.eval_dataset)):
            verifier_tokenisation(ds, donnees[split], tokenizer, 4096)
        protocole = {'format_sft': FORMAT_SFT, 'arguments': vars(args) | {'revision': args.revision},
            'configuration_dpo': config.to_dict(), 'manifeste_donnees': manifeste,
            'reference': 'copie_gelee_du_lora_sft_initial', 'reference_sha256': empreinte_initiale,
            'adapter_source_sha256': empreinte(args.adapter / 'adapter_model.safetensors'),
            'scripts_sha256': {p.name: empreinte(p) for p in (Path(__file__), RACINE / 'training/preparer_dpo.py')},
            'versions': {p: version(p) for p in ('trl', 'transformers', 'peft', 'datasets', 'torch')}}
        chemin = args.output / 'protocole_entrainement.json'
        chemin.write_text(json.dumps(protocole, default=str, ensure_ascii=False, indent=2))
        mlflow.log_params({'revision': args.revision, 'initial_adapter': str(args.adapter),
            'beta': 0.1, 'learning_rate': 5e-6, 'max_steps': 32, 'batch_effectif': 16,
            'train': len(donnees['train']), 'validation': len(donnees['validation']),
            'reference_sha256': empreinte_initiale, 'format_sft': FORMAT_SFT})
        mlflow.log_artifact(str(chemin))
        avant = trainer.evaluate()
        resultat = trainer.train()
        apres = trainer.evaluate()
        assert empreinte_adaptateur(modele, 'ref') == empreinte_initiale, 'La référence a changé.'
        assert empreinte_adaptateur(modele, 'default') != empreinte_initiale, 'Le LoRA n’a pas changé.'
        modele.save_pretrained(args.output / 'adapter', selected_adapters=['default'])
        tokenizer.save_pretrained(args.output / 'adapter')
        trainer.save_state()
        resume = {'mlflow_run_id': run.info.run_id, 'duree_secondes': time.perf_counter() - debut,
            'avant': avant, 'apres': apres, 'train': resultat.metrics,
            'best_model_checkpoint': trainer.state.best_model_checkpoint,
            'reference_inchangee': True, 'tokenisation_verifiee': True}
        chemin_resume = args.output / 'resume_entrainement.json'
        chemin_resume.write_text(json.dumps(resume, ensure_ascii=False, indent=2))
        mlflow.log_artifact(str(chemin_resume))
        print(json.dumps(resume, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
