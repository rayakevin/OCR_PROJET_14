"""Met à jour le Space existant après contrôle des poids, puis vérifie son démarrage."""

import argparse
import json
from pathlib import Path
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download
from serving.space.publier import preparer

SPACE = 'rayakevin/ocr14-triage-fr-en'
# Version publique contenant l'adaptateur sélectionné, indépendant du code à déployer.
REVISION_POIDS = 'f32b2a65f9ef05fb431b4e07d65253d82b5d049a'


def main():
    """Ne publie que le paquet autorisé et refuse tout matériel payant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preuve', type=Path, default=Path('publication.json'))
    args = parser.parse_args()
    api = HfApi()
    runtime = api.get_space_runtime(SPACE)
    if runtime.raw['hardware']['requested'] != 'zero-a10g':
        raise RuntimeError('Déploiement réservé au Space ZeroGPU gratuit existant.')
    parent = api.space_info(SPACE).sha
    with tempfile.TemporaryDirectory() as dossier:
        racine = Path(dossier)
        for nom in ('adapter_config.json', 'adapter_model.safetensors'):
            hf_hub_download(SPACE, f'adapter/{nom}', repo_type='space',
                            revision=REVISION_POIDS, local_dir=racine / 'poids')
        empreintes = preparer(racine / 'poids/adapter', racine / 'paquet')
        commit = api.upload_folder(
            repo_id=SPACE, repo_type='space', folder_path=racine / 'paquet',
            parent_commit=parent, commit_message='Déploiement vérifié depuis le dépôt OCR14')
    # Le SHA du runtime évite de confondre l'ancienne instance avec la nouvelle.
    for _ in range(80):
        runtime = api.get_space_runtime(SPACE)
        if runtime.stage == 'RUNNING' and runtime.raw.get('sha') == commit.oid:
            break
        if runtime.stage in ('BUILD_ERROR', 'RUNTIME_ERROR', 'CONFIG_ERROR'):
            raise RuntimeError(f'Échec du Space : {runtime.stage}')
        print('Space :', runtime.stage, flush=True)
        time.sleep(10)
    else:
        raise TimeoutError('Le nouveau commit ne fonctionne pas encore après le délai prévu.')
    preuve = {'space': SPACE, 'commit': commit.oid, 'stage': str(runtime.stage),
              'revision_poids': REVISION_POIDS, 'fichiers_sha256': empreintes,
              'limite': 'Démarrage vérifié ; aucune inférence clinique lancée par le déploiement.'}
    args.preuve.parent.mkdir(parents=True, exist_ok=True)
    args.preuve.write_text(json.dumps(preuve, ensure_ascii=False, indent=2) + '\n')
    print('Space démarré sur le commit', commit.oid)


if __name__ == '__main__':
    main()
