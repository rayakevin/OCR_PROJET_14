"""Je vérifie les poids avant de servir le LoRA retenu avec vLLM."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

MANIFESTE = json.loads(Path(__file__).with_name('modele.json').read_text())


def commande(adaptateur):
    """Refuse un adaptateur différent de celui évalué ; retourne la commande sans secret."""
    poids = adaptateur / 'adapter_model.safetensors'
    if hashlib.sha256(poids.read_bytes()).hexdigest() != MANIFESTE['adapter_sha256']:
        raise ValueError('Les poids ne correspondent pas au SFT mixte époque 3 sélectionné.')
    config = json.loads((adaptateur / 'adapter_config.json').read_text())
    rang = config['r']
    if rang not in (8, 16, 32, 64, 128, 256):
        raise ValueError('Rang LoRA non prévu pour ce POC.')
    return ['vllm', 'serve', MANIFESTE['socle'], '--revision', MANIFESTE['revision'],
            '--tokenizer-revision', MANIFESTE['revision'], '--dtype', MANIFESTE['dtype'],
            '--port', '8000', '--host', '0.0.0.0', '--max-model-len', '2240',
            '--generation-config', 'vllm', '--enable-lora', '--max-lora-rank', str(rang),
            '--lora-modules', f'{MANIFESTE["id"]}={adaptateur.resolve()}',
            '--gpu-memory-utilization', '0.80', '--no-enable-log-requests']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('adaptateur', type=Path)
    args = parser.parse_args()
    if version('vllm') != '0.32.0':
        raise RuntimeError('Ce lancement est préparé pour vLLM 0.32.0.')
    if len(os.environ.get('VLLM_API_KEY', '')) < 24:
        raise RuntimeError('Définir VLLM_API_KEY avec au moins 24 caractères.')
    subprocess.run(commande(args.adaptateur), check=True)


if __name__ == '__main__':
    main()
