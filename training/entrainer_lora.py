"""J'entraîne un adaptateur LoRA SFT et je trace le run dans MLflow."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from importlib.metadata import version

import mlflow
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from training.format_sft import FORMAT_SFT, encoder_exemple
from training.continuer_mixte import verifier_depart
DONNEES = RACINE / "data/processed/sft_deux_phases"
MODELE_BASE = "Qwen/Qwen3-1.7B-Base"
SEED = 3407


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def empreinte(chemin):
    """Calcule l'empreinte SHA-256 d'un fichier utilisé par le run."""
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


class CorpusSFT(Dataset):
    """Pré-encode un corpus afin de ne pas retokeniser à chaque époque."""

    def __init__(self, tokenizer, exemples, longueur_max):
        self.lignes = [encoder_exemple(tokenizer, x, longueur_max) for x in exemples]

    def __len__(self):
        return len(self.lignes)

    def __getitem__(self, index):
        return self.lignes[index]


class CollateurSFT:
    """Applique un padding dynamique et conserve le masque des labels."""

    def __init__(self, pad_token_id, multiple=8):
        self.pad_token_id = pad_token_id
        self.multiple = multiple

    def __call__(self, exemples):
        longueur = max(len(x["input_ids"]) for x in exemples)
        longueur = ((longueur + self.multiple - 1) // self.multiple) * self.multiple
        ids, labels, masques = [], [], []
        for exemple in exemples:
            marge = longueur - len(exemple["input_ids"])
            ids.append(exemple["input_ids"] + [self.pad_token_id] * marge)
            labels.append(exemple["labels"] + [-100] * marge)
            masques.append([1] * len(exemple["input_ids"]) + [0] * marge)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masques, dtype=torch.long),
        }


class JournalMLflow(TrainerCallback):
    """Enregistre dans le run actif les métriques émises par Trainer."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        mesures = {
            cle: float(valeur)
            for cle, valeur in (logs or {}).items()
            if isinstance(valeur, (int, float)) and cle != "epoch"
        }
        if mesures and mlflow.active_run():
            mlflow.log_metrics(mesures, step=int(state.global_step))


def analyser_arguments():
    """Définit les paramètres utiles au smoke test et au run complet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELE_BASE)
    parser.add_argument("--revision", required=True, help="Révision figée du socle et du tokenizer.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--phase", choices=("phase1_qa", "phase2_triage", "sft_mixte"), default="phase1_qa")
    parser.add_argument("--adapter", type=Path, help="Adaptateur de phase 1 à poursuivre en phase 2.")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{RACINE / 'mlflow.db'}"))
    parser.add_argument("--experiment", default="OCR_PROJET_14-SFT")
    parser.add_argument("--artifact-location", help="Dossier des artefacts pour une nouvelle expérience.")
    parser.add_argument("--train", type=Path, default=DONNEES / "phase1_qa_train.jsonl")
    parser.add_argument("--validation", type=Path, default=DONNEES / "phase1_qa_validation.jsonl")
    parser.add_argument("--output", type=Path, default=RACINE / "models/phase1_qa_lora")
    parser.add_argument("--run-name", default="phase1-qa-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, help="Fixe le warmup pour comparer des runs à budget égal.")
    parser.add_argument("--checkpoint-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="Nombre de checkpoints conservés pour la comparaison sur validation.")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Poursuit les poids du SFT mixte avec un nouvel optimiseur et calendrier.")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    """Charge les données, entraîne l'adaptateur et sauvegarde le meilleur état."""
    args = analyser_arguments()
    if args.phase == "sft_mixte" and args.adapter:
        raise ValueError("Le SFT mixte repart du socle, sans adaptateur initial.")
    if args.initial_checkpoint and (args.phase != "sft_mixte" or args.adapter or args.resume_from_checkpoint):
        raise ValueError("La poursuite mixte ne se combine pas avec --adapter ou --resume-from-checkpoint.")
    depart = verifier_depart(args.initial_checkpoint, args.train, args.validation, args.revision) if args.initial_checkpoint else None
    adaptateur_initial = args.initial_checkpoint or args.adapter
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Le dossier de sortie contient déjà un run. Choisir un nouveau dossier.")
    args.output.mkdir(parents=True, exist_ok=True)

    train = lire_jsonl(args.train)
    validation = lire_jsonl(args.validation)
    if args.train_limit:
        train = random.Random(args.seed).sample(train, min(args.train_limit, len(train)))
    if args.validation_limit:
        validation = random.Random(args.seed).sample(validation, min(args.validation_limit, len(validation)))

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    corpus_train = CorpusSFT(tokenizer, train, args.max_length)
    corpus_validation = CorpusSFT(tokenizer, validation, args.max_length)

    protocole = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "format_sft": FORMAT_SFT, "eos_token_id": tokenizer.eos_token_id,
        "train_ids": [x["id"] for x in train], "validation_ids": [x["id"] for x in validation],
        "max_tokens_train": max(x["length"] for x in corpus_train.lignes),
        "max_tokens_validation": max(x["length"] for x in corpus_validation.lignes),
        "train_sha256": empreinte(args.train), "validation_sha256": empreinte(args.validation),
        "scripts_sha256": {str(p.relative_to(RACINE)): empreinte(p) for p in (
            Path(__file__), RACINE / "training/format_sft.py")},
        "versions": {nom: version(nom) for nom in ("torch", "transformers", "peft", "mlflow")},
    }
    if depart:
        protocole["poursuite"] = depart
    chemin_protocole = args.output / "protocole_entrainement.json"
    if args.resume_from_checkpoint:
        precedent = json.loads(chemin_protocole.read_text())
        # Seul le chemin de reprise peut changer entre deux morceaux du même run.
        for document in (precedent, protocole):
            document["arguments"].pop("resume_from_checkpoint", None)
        if precedent != protocole:
            raise ValueError("Le protocole du checkpoint diffère ; je refuse une reprise incompatible.")
    else:
        chemin_protocole.write_text(json.dumps(protocole, ensure_ascii=False, indent=2) + "\n")

    modele = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype),
    )
    modele.config.use_cache = False
    modele.gradient_checkpointing_enable()
    modele.enable_input_require_grads()
    configuration_lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if adaptateur_initial:
        dossier_origine = (args.initial_checkpoint.parent.parent if args.initial_checkpoint else args.adapter.parent)
        origine = json.loads((dossier_origine / "protocole_entrainement.json").read_text())
        if origine["format_sft"] != FORMAT_SFT or origine["arguments"]["revision"] != args.revision:
            raise ValueError("L'adaptateur initial doit utiliser le même format et le même socle.")
        modele = PeftModel.from_pretrained(modele, str(adaptateur_initial), is_trainable=True)
        configuration_lora = modele.peft_config["default"]
    else:
        modele = get_peft_model(modele, configuration_lora)
    entrainables, total = modele.get_nb_trainable_parameters()

    parametres = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=(args.warmup_steps if args.warmup_steps is not None
                      else 20 if args.max_steps < 0 else min(2, args.max_steps)),
        weight_decay=0.01,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        gradient_checkpointing=True,
        use_cache=False,
        eval_strategy="steps",
        eval_steps=(args.checkpoint_steps if args.max_steps < 0
                    else min(args.checkpoint_steps, args.max_steps)),
        save_strategy="steps",
        save_steps=(args.checkpoint_steps if args.max_steps < 0
                    else min(args.checkpoint_steps, args.max_steps)),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5 if args.max_steps < 0 else 1,
        logging_first_step=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(args.experiment) is None:
        client.create_experiment(args.experiment, artifact_location=args.artifact_location)
    mlflow.set_experiment(args.experiment)
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    debut = time.perf_counter()
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "model": args.model,
            "phase": args.phase,
            "revision": args.revision,
            "dtype": args.dtype,
            "format_sft": FORMAT_SFT,
            "initial_adapter": str(adaptateur_initial) if adaptateur_initial else "aucun",
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_sha256": empreinte(args.train),
            "validation_sha256": empreinte(args.validation),
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "seed": args.seed,
            "checkpoint_steps": args.checkpoint_steps,
            "save_total_limit": args.save_total_limit,
            "warmup_steps": parametres.warmup_steps,
            "lora_r": configuration_lora.r,
            "lora_alpha": configuration_lora.lora_alpha,
            "lora_dropout": configuration_lora.lora_dropout,
            "trainable_parameters": entrainables,
            "total_parameters": total,
            "output_dir": str(args.output),
        })
        if depart:
            mlflow.log_params({f"poursuite_{k}": v for k, v in depart.items()})
        mlflow.log_artifact(str(chemin_protocole), artifact_path="configuration")
        trainer = Trainer(
            model=modele,
            args=parametres,
            train_dataset=corpus_train,
            eval_dataset=corpus_validation,
            data_collator=CollateurSFT(tokenizer.pad_token_id),
            processing_class=tokenizer,
            callbacks=[JournalMLflow()],
        )
        resultat = trainer.train(
            resume_from_checkpoint=(
                str(args.resume_from_checkpoint.resolve())
                if args.resume_from_checkpoint else None
            )
        )
        evaluation = trainer.evaluate()
        modele.save_pretrained(args.output / "adapter")
        tokenizer.save_pretrained(args.output / "adapter")
        trainer.save_state()
        duree = time.perf_counter() - debut
        resume = {
            "mlflow_run_id": run.info.run_id,
            "duree_secondes": round(duree, 2),
            "train_metrics": resultat.metrics,
            "validation_metrics": evaluation,
            "trainable_parameters": entrainables,
            "total_parameters": total,
        }
        (args.output / "resume_entrainement.json").write_text(
            json.dumps(resume, ensure_ascii=False, indent=2) + "\n"
        )
        mlflow.log_metrics({
            "duree_secondes": duree,
            **{f"final_{k}": float(v) for k, v in evaluation.items() if isinstance(v, (int, float))},
        })
        mlflow.log_artifact(str(args.output / "resume_entrainement.json"), artifact_path="resume")
        mlflow.log_artifact(str(args.output / "adapter/adapter_config.json"), artifact_path="configuration")
        print(json.dumps(resume, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
