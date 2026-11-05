"""Entrée du Space : l'import spaces précède impérativement PyTorch."""

import spaces

import hashlib
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from serving.space.interface import creer_interface
from serving.space.parcours import MODELE
from training.format_sft import construire_prompt

ADAPTATEUR = Path('adapter')
if hashlib.sha256((ADAPTATEUR / 'adapter_model.safetensors').read_bytes()).hexdigest() != MODELE['adapter_sha256']:
    raise ValueError('Adaptateur différent du SFT v3 évalué.')

tokenizer = AutoTokenizer.from_pretrained(MODELE['socle'], revision=MODELE['revision'])
if tokenizer.eos_token_id != MODELE['eos_token_id']:
    raise ValueError('EOS différent de celui utilisé pendant l’évaluation.')
socle = AutoModelForCausalLM.from_pretrained(
    MODELE['socle'], revision=MODELE['revision'], dtype=torch.bfloat16,
    attn_implementation='sdpa',
)
# PEFT déduit sinon « cuda » et safetensors demande un vrai GPU pendant le démarrage.
# Je lis donc le LoRA sur CPU, puis laisse ZeroGPU préparer son transfert hors inférence.
modele = PeftModel.from_pretrained(socle, ADAPTATEUR, torch_device='cpu').eval().to('cuda')


@spaces.GPU(duration=40)
def generer_gpu(ids):
    """Génère sans échantillonnage ; aucun entraînement ni fusion des poids."""
    with torch.inference_mode():
        entree = torch.tensor([ids], device='cuda')
        sortie = modele.generate(
            input_ids=entree, attention_mask=torch.ones_like(entree),
            generation_config=GenerationConfig(
                do_sample=False, max_new_tokens=MODELE['max_new_tokens'],
                eos_token_id=MODELE['eos_token_id'], pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            ),
        )
        return sortie[0, len(ids):].tolist()


def generer(cas, langue):
    """Valide le prompt sur CPU avant de demander du temps GPU."""
    prompt = construire_prompt(tokenizer, [
        {'role': 'system', 'content': MODELE['systemes'][langue]}, {'role': 'user', 'content': cas}])
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > MODELE['max_input_tokens']:
        raise ValueError('Entrée trop longue ; aucune troncature automatique.')
    sortie = generer_gpu(ids)
    return {'texte': tokenizer.decode(sortie, skip_special_tokens=True).strip(),
            'fin': 'stop' if MODELE['eos_token_id'] in sortie else 'length',
            'usage': {'prompt_tokens': len(ids), 'completion_tokens': len(sortie)},
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}


if __name__ == '__main__':
    creer_interface(generer).launch(
        server_name='0.0.0.0', server_port=7860, show_error=False,
        blocked_paths=[str(ADAPTATEUR.resolve())],
    )
