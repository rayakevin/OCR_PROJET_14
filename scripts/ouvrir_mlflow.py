"""J'ouvre un suivi MLflow local dans le navigateur, sans exposer de port public."""

import argparse
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

RACINE = Path(__file__).resolve().parents[1]
ARCHIVE = RACINE / 'data/colab_archive'
BASES = {
    'historique': ARCHIVE / 'local/mlflow_colab.db',
    'final': ARCHIVE / 'local/mlflow_finalisation.db',
    'mixte': ARCHIVE / 'sft_mixte_20261106_091447/mlflow.db',
    'poursuite': ARCHIVE / 'poursuite_mixte_20261106_095443/mlflow.db',
    'dpo-cible': ARCHIVE / 'dpo_triage_cible_20261106_132004/mlflow.db',
    'local': RACINE / 'mlflow.db',
}


def copie_consultation(source):
    """Préserve l'archive et adapte les URI Drive dans une copie régénérable.

    Les métriques sont dans SQLite ; un lien d'artefact n'est utilisable que si
    le fichier a aussi été récupéré. Je ne fabrique pas les artefacts absents.
    """
    cible = ARCHIVE / 'consultation' / source.parent.name / source.name
    cible.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(cible) as copie:
            original.backup(copie)
            for table, colonne in (('runs', 'artifact_uri'), ('experiments', 'artifact_location')):
                copie.execute(f'UPDATE {table} SET {colonne}=replace({colonne}, ?, ?)',
                              ('file:///content/drive/MyDrive/OCR_PROJET_14/experiences/' + source.parent.name,
                               source.parent.as_uri()))
            if copie.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise RuntimeError('Base MLflow invalide.')
    return cible


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experience', choices=BASES, default='mixte')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--sans-navigateur', action='store_true')
    args = p.parse_args()
    source = BASES[args.experience]
    if not source.is_file():
        raise FileNotFoundError(f'Archive à récupérer : {source}')
    # Je refuse de remplacer un serveur déjà présent sur ce port.
    with socket.socket() as test:
        test.bind(('127.0.0.1', args.port))
    base = copie_consultation(source)
    url = f'http://127.0.0.1:{args.port}'
    commande = [sys.executable, '-m', 'mlflow', 'ui', '--backend-store-uri', f'sqlite:///{base}',
                '--host', '127.0.0.1', '--port', str(args.port)]
    print(f'MLflow : {args.experience} — {url}\nArrêt : Ctrl+C. Les archives originales restent intactes.', flush=True)
    processus = subprocess.Popen(commande, cwd=RACINE)
    try:
        for _ in range(60):
            if processus.poll() is not None:
                raise RuntimeError('Le serveur MLflow ne démarre pas ; consulter le journal ci-dessus.')
            try:
                with urlopen(url + '/health', timeout=1) as reponse:
                    if reponse.status == 200:
                        if not args.sans_navigateur:
                            webbrowser.open(url)
                        break
            except (URLError, TimeoutError):
                time.sleep(.5)
        else:
            raise TimeoutError('MLflow ne répond pas après 30 secondes.')
        processus.wait()
    except KeyboardInterrupt:
        pass
    finally:
        processus.terminate()
        try:
            processus.wait(timeout=10)
        except subprocess.TimeoutExpired:
            processus.kill()
            processus.wait()


if __name__ == '__main__':
    main()
