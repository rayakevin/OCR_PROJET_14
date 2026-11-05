"""Je vérifie le contrat, les traces et la latence d'un endpoint réel sur trois cas fictifs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

import httpx

CAS_FICTIFS = [
    {'langue': 'fr', 'plainte': 'Cas fictif : un adulte décrit une gêne à la gorge.',
     'reponses': {'debut': 'Deux jours', 'evolution': 'Stable', 'contexte': 'Âge 35 ans ; antécédents inconnus'}},
    {'langue': 'en', 'plainte': 'Fictional case: an adult reports a sore throat.',
     'reponses': {'debut': 'Two days', 'evolution': 'Stable', 'contexte': 'Age 35; medical history unknown'}},
    {'langue': 'fr', 'plainte': 'Cas fictif : une personne décrit une douleur au poignet.',
     'reponses': {'debut': 'Ce matin', 'evolution': 'Stable', 'contexte': 'Âge 40 ans ; antécédents inconnus',
                  'retentissement': 'Gêne pour écrire'}},
]


def verifier(url, cle):
    """Mesure l'aller-retour HTTP et vérifie que la trace correspond à chaque réponse."""
    lignes = []
    with httpx.Client(base_url=url.rstrip('/') + '/', timeout=75,
                      headers={'Authorization': f'Bearer {cle}'}) as client:
        client.get('readyz').raise_for_status()
        for cas in CAS_FICTIFS:
            debut = time.perf_counter()
            reponse = client.post('v1/consultations', json=cas)
            millisecondes = 1000 * (time.perf_counter() - debut)
            reponse.raise_for_status()
            document = reponse.json()
            assert document['statut'] in ('proposition_a_valider', 'revue_humaine')
            trace = client.get('v1/interactions/' + document['interaction_id'])
            trace.raise_for_status()
            assert trace.json()['resultat'] == document
            lignes.append({'interaction_id': document['interaction_id'], 'langue': cas['langue'],
                           'statut': document['statut'], 'latence_http_ms': round(millisecondes, 2),
                           'latence_api_ms': document['latence_ms'],
                           'usage': document['inference']['usage'], 'modele': document['modele']})
    # Le premier appel comprend éventuellement le chargement du tokenizer.
    return {'date_utc': datetime.now(timezone.utc).isoformat(), 'cas': 'fictifs_sans_labels_cliniques',
            'mesures': lignes, 'latence_mediane_ms': statistics.median(x['latence_http_ms'] for x in lignes),
            'limite': 'Trois requêtes séquentielles ; ni test de charge ni mesure de pertinence clinique.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choisir un nouveau rapport pour conserver la mesure précédente.')
    rapport = verifier(args.url, os.environ['OCR14_API_KEY'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(rapport, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
