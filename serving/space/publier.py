"""Je prépare un paquet explicite, puis publie uniquement sur ZeroGPU gratuit."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from serving.space.parcours import MODELE

RACINE = Path(__file__).resolve().parents[2]
# Liste fermée : ni .env, ni corpus, ni résultats individuels, ni journaux.
FICHIERS = ('serving/__init__.py', 'serving/domaine.py', 'serving/modele.json',
            'serving/space/__init__.py', 'serving/space/interface.py',
            'serving/space/parcours.py', 'training/format_sft.py',
            'serving/space/support.py', 'serving/space/support/index.html',
            'serving/space/support/resultats.json', 'output/pdf/rapport_technique_ocr14.pdf')


def preparer(adaptateur, destination):
    """Vérifie le LoRA retenu et refuse de mélanger le paquet à un dossier existant."""
    poids = adaptateur / 'adapter_model.safetensors'
    if hashlib.sha256(poids.read_bytes()).hexdigest() != MODELE['adapter_sha256']:
        raise ValueError('Poids différents du SFT v3 sélectionné.')
    configuration = json.loads((adaptateur / 'adapter_config.json').read_text())
    if configuration['base_model_name_or_path'] != MODELE['socle']:
        raise ValueError('Socle de l’adaptateur différent du manifeste.')
    destination.mkdir(parents=True, exist_ok=False)
    for nom in FICHIERS:
        cible = destination / nom
        cible.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RACINE / nom, cible)
    for nom in ('app.py', 'requirements.txt', 'README.md'):
        shutil.copyfile(RACINE / 'serving/space' / nom, destination / nom)
    (destination / 'adapter').mkdir()
    for nom in ('adapter_config.json', 'adapter_model.safetensors'):
        shutil.copyfile(adaptateur / nom, destination / 'adapter' / nom)
    return {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--publier', metavar='COMPTE/SPACE', help='Création publique puis envoi du paquet.')
    args = parser.parse_args()
    empreintes = preparer(args.adapter, args.destination)
    print(f'Paquet vérifié : {len(empreintes)} fichiers.')
    if not args.publier:
        return
    from huggingface_hub import HfApi, SpaceHardware
    api = HfApi()
    # Aucun repli automatique vers une machine payante. Refus si le dépôt existe déjà.
    url = api.create_repo(args.publier, repo_type='space', space_sdk='gradio',
                          space_hardware=SpaceHardware.ZERO_A10G, private=False, exist_ok=False)
    print(url)
    commit = api.upload_folder(repo_id=args.publier, repo_type='space', folder_path=args.destination,
                               commit_message='Démonstration SFT triage v3 avec Gradio et ZeroGPU')
    preuve = {'url': str(url), 'commit': commit.oid, 'fichiers_sha256': empreintes,
              'hardware_demande': 'zero-a10g', 'inference_verifiee': False}
    args.destination.with_suffix('.publication.json').write_text(
        json.dumps(preuve, ensure_ascii=False, indent=2) + '\n')
    print('Publication enregistrée ; démarrage et inférence restent à vérifier.')


if __name__ == '__main__':
    main()
