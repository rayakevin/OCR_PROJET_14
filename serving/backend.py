"""Client vLLM : je conserve les tokens du prompt validé pendant l'évaluation."""

import hashlib
import json
import time
from pathlib import Path
from threading import Lock

import httpx
from training.format_sft import construire_prompt
from serving.traces_http import tracer_sortie

MODELE = json.loads(Path(__file__).with_name('modele.json').read_text())


class EntreeTropLongue(ValueError):
    """L'entrée dépasse le contexte autorisé ; aucune information n'est tronquée."""


class Vllm:
    """Utilise /completions pour éviter qu'un second template modifie le préfixe SFT."""
    def __init__(self, url, cle):
        self.client = httpx.Client(base_url=url.rstrip('/') + '/', timeout=60,
                                   headers={'Authorization': f'Bearer {cle}'})
        self.tokenizer = None
        self.verrou = Lock()

    def disponible(self):
        reponse = self.requete('GET', 'models')
        return MODELE['id'] in {m['id'] for m in reponse.json()['data']}

    def requete(self, methode, route, **options):
        """Trace statut et durée, y compris timeout/erreur HTTP, sans copier le prompt."""
        debut = time.perf_counter()
        statut, erreur_type = None, None
        try:
            reponse = self.client.request(methode, route, **options)
            statut = reponse.status_code
            reponse.raise_for_status()
            return reponse
        except Exception as erreur:
            erreur_type = type(erreur).__name__
            raise
        finally:
            tracer_sortie(f'{methode} /v1/{route}', statut,
                          round(1000 * (time.perf_counter() - debut), 2), erreur_type)

    def generer(self, cas, langue):
        """Refuse les entrées longues puis envoie les IDs exacts, avec décodage glouton."""
        with self.verrou:
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(MODELE['socle'], revision=MODELE['revision'])
        prompt = construire_prompt(self.tokenizer, [
            {'role': 'system', 'content': MODELE['systemes'][langue]}, {'role': 'user', 'content': cas}])
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MODELE['max_input_tokens']:
            raise EntreeTropLongue('Entrée trop longue ; reformuler sans supprimer un élément utile.')
        assert self.tokenizer.eos_token_id == MODELE['eos_token_id']
        reponse = self.requete('POST', 'completions', json={
            'model': MODELE['id'], 'prompt': ids, 'temperature': 0, 'max_tokens': MODELE['max_new_tokens'],
            'stop_token_ids': [MODELE['eos_token_id']], 'skip_special_tokens': True,
            'seed': 3407, 'n': 1,
        })
        document = reponse.json()
        choix = document['choices'][0]
        if not isinstance(choix['text'], str):
            raise ValueError('Réponse vLLM incorrecte.')
        return {'texte': choix['text'].strip(), 'fin': choix['finish_reason'],
                'usage': document.get('usage', {}),
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}

    def fermer(self):
        self.client.close()
