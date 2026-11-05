"""Je démarre vLLM et l'API dans une VM Colab, mesure trois cas, puis les arrête."""

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

import httpx

from serving.verifier_endpoint import verifier


def attendre(url, cle, processus, delai):
    """Attend une réponse authentifiée, en détectant aussi l'arrêt du serveur."""
    limite = time.monotonic() + delai
    while time.monotonic() < limite:
        if processus.poll() is not None:
            raise RuntimeError(f'Serveur arrêté : code {processus.returncode}. Consulter son journal.')
        try:
            reponse = httpx.get(url, headers={'Authorization': f'Bearer {cle}'}, timeout=5)
            if reponse.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(5)
    raise TimeoutError('Le serveur ne devient pas disponible dans le délai prévu.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    racine = Path(__file__).resolve().parents[1]
    environnement = {**os.environ, 'PYTHONUNBUFFERED': '1',
        'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
        'OCR14_API_KEY': secrets.token_urlsafe(32), 'VLLM_API_KEY': secrets.token_urlsafe(32),
        'VLLM_URL': 'http://127.0.0.1:8000/v1',
        'OCR14_AUDIT_PATH': str((args.output / 'interactions.sqlite3').resolve())}
    processus = []
    try:
        with (args.output / 'vllm.log').open('w') as log:
            moteur = subprocess.Popen([sys.executable, '-m', 'serving.lancer_vllm', str(args.adapter)],
                cwd=racine, env=environnement, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processus.append(moteur)
        attendre('http://127.0.0.1:8000/v1/models', environnement['VLLM_API_KEY'], moteur, 900)
        # Colab peut déjà occuper 8080. Je réserve un port libre et transmets
        # la socket à Uvicorn pour éviter qu'un autre processus ne le prenne.
        with socket.socket() as ecoute, (args.output / 'api.log').open('w') as log:
            ecoute.bind(('127.0.0.1', 0))
            ecoute.listen(128)
            url_api = f'http://127.0.0.1:{ecoute.getsockname()[1]}'
            api = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'serving.api:app',
                '--fd', str(ecoute.fileno()), '--no-access-log'], cwd=racine,
                env=environnement, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(ecoute.fileno(),), start_new_session=True)
        processus.append(api)
        attendre(url_api + '/readyz', environnement['OCR14_API_KEY'], api, 60)
        rapport = verifier(url_api, environnement['OCR14_API_KEY'])
        # Ces contrôles portent sur le contrat de l'API réelle, pas sur la pertinence médicale.
        sans_cle = httpx.get(url_api + '/readyz').status_code
        with httpx.Client(base_url=url_api,
                headers={'Authorization': 'Bearer ' + environnement['OCR14_API_KEY']}) as client:
            incomplet = client.post('/v1/consultations', json={'langue': 'fr', 'plainte': 'Gêne fictive'}).json()
            invalide = client.post('/v1/consultations', json={'langue': 'xx'}).status_code
        assert sans_cle == 401 and invalide == 422 and incomplet['statut'] == 'a_completer'
        rapport['controles_http'] = {'sans_cle': sans_cle, 'entree_invalide': invalide,
                                    'cas_incomplet': incomplet['statut']}
        rapport['hebergement'] = 'VM Colab temporaire ; appels HTTP internes, sans URL publique persistante.'
        (args.output / 'verification_endpoint.json').write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(rapport, ensure_ascii=False, indent=2), flush=True)
    finally:
        # Les workers CUDA sont dans le même groupe ; arrêter seulement le parent les laisserait actifs.
        for p in reversed(processus):
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


if __name__ == '__main__':
    main()
