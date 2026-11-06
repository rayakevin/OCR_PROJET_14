"""Expose temporairement FastAPI/vLLM pour l'oral, puis arrête tous les processus.

Seule l'API est exposée par le tunnel HTTPS. Les consultations et traces exigent
la clé Bearer ; les requêtes de vérification utilisent exclusivement des cas fictifs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time

from serving.verifier_colab import attendre
from serving.verifier_endpoint import verifier


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--cloudflared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minutes',type=int,default=20)
    a=p.parse_args()
    if not 1 <= a.minutes <= 60:raise ValueError('Durée autorisée : 1 à 60 minutes.')
    if len(os.environ.get('OCR14_API_KEY',''))<24:raise ValueError('Clé OCR14_API_KEY trop courte.')
    a.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONUNBUFFERED':'1',
         'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
         'VLLM_API_KEY':secrets.token_urlsafe(32),'VLLM_URL':'http://127.0.0.1:8000/v1',
         'OCR14_AUDIT_PATH':str((a.output/'interactions.sqlite3').resolve())}
    processus=[]
    def lancer(cmd,log,**options):
        with (a.output/log).open('w') as f:
            proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,
                                  start_new_session=True,**options)
        processus.append(proc)
        return proc
    statut={'etat':'demarrage','moteur':'vllm','date_utc':datetime.now(timezone.utc).isoformat()}
    try:
        print('Démarrage de vLLM (chargement initial possible de plusieurs minutes).',flush=True)
        moteur=lancer([sys.executable,'-m','serving.lancer_vllm',str(a.adapter)],'vllm.log')
        attendre('http://127.0.0.1:8000/v1/models',env['VLLM_API_KEY'],moteur,900)
        with socket.socket() as ecoute:
            ecoute.bind(('127.0.0.1',0));ecoute.listen(128)
            local=f'http://127.0.0.1:{ecoute.getsockname()[1]}'
            api=lancer([sys.executable,'-m','uvicorn','serving.api:app','--fd',str(ecoute.fileno()),
                        '--no-access-log'],'api.log',pass_fds=(ecoute.fileno(),))
        attendre(local+'/readyz',env['OCR14_API_KEY'],api,60)
        tunnel=lancer([str(a.cloudflared),'tunnel','--url',local,'--no-autoupdate'],'tunnel.log')
        url=None
        for _ in range(60):
            if tunnel.poll() is not None:raise RuntimeError('Tunnel arrêté ; consulter tunnel.log.')
            texte=(a.output/'tunnel.log').read_text(errors='replace')
            urls=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',texte)
            if urls:url=urls[-1];break
            time.sleep(1)
        if not url:raise TimeoutError('Pas d’URL de tunnel après 60 secondes.')
        attendre(url+'/readyz',env['OCR14_API_KEY'],tunnel,120)
        rapport=verifier(url,env['OCR14_API_KEY'])
        rapport.update(url=url,hebergement='Colab + tunnel HTTPS temporaire ; API authentifiée')
        (a.output/'verification_publique.json').write_text(json.dumps(rapport,ensure_ascii=False,indent=2)+'\n')
        statut.update(etat='verifie_public',url=url,minutes=a.minutes)
        (a.output/'statut.json').write_text(json.dumps(statut,ensure_ascii=False,indent=2)+'\n')
        print(f'DEMO_URL={url}\nVérification HTTPS réussie. Utiliser la clé saisie au lancement dans la page.',flush=True)
        print(f'Arrêt automatique dans {a.minutes} minutes. Interrompre la cellule pour arrêter plus tôt.',flush=True)
        limite=time.monotonic()+a.minutes*60
        while time.monotonic()<limite:
            if any(proc.poll() is not None for proc in processus):raise RuntimeError('Un service s’est arrêté.')
            time.sleep(2)
        statut['etat']='termine_apres_verification'
    except BaseException as erreur:
        statut.update(etat='interrompu' if isinstance(erreur,KeyboardInterrupt) else 'echec',
                      erreur_type=type(erreur).__name__)
        raise
    finally:
        for proc in reversed(processus):
            try:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=20)
            except ProcessLookupError:pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        (a.output/'statut.json').write_text(json.dumps(statut,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
