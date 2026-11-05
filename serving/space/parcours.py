"""Je conserve le questionnaire et sa trace dans la session du visiteur."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from serving.domaine import Consultation, interpreter, presentation, prochaine_question

MODELE = json.loads(Path(__file__).resolve().parents[1].joinpath('modele.json').read_text())


def nouvelle_session(langue):
    """Crée un état indépendant ; changer de langue commence un nouveau cas."""
    cas = Consultation(langue=langue, session_id=uuid4())
    return {'cas': cas.model_dump(mode='json'), 'evenements': [], 'termine': False}


def avancer(texte, etat, generer, mode='guide'):
    """Collecte un champ puis appelle le modèle seulement si aucun champ ne manque.

    Le générateur est injecté : les tests du parcours n'ont besoin ni de poids
    ni de GPU. Une erreur d'inférence reste une abstention explicite et tracée.
    """
    if mode not in ('guide', 'complet'):
        raise ValueError('Mode de saisie inconnu.')
    suivant = deepcopy(etat)
    if suivant['termine']:
        return suivant  # Un double clic ne consomme pas un deuxième appel GPU.
    cas = Consultation.model_validate(suivant['cas'])
    question = prochaine_question(cas)
    if not texte.strip():
        raise ValueError('Répondre à la question ou indiquer que l’information est inconnue.')
    donnees = cas.model_dump(mode='json')
    if question['champ'] == 'plainte':
        donnees['plainte'] = texte
    else:
        donnees['reponses'][question['champ']] = texte
    cas = Consultation.model_validate(donnees)
    suivant['cas'] = cas.model_dump(mode='json')
    evenement = {'interaction_id': str(uuid4()), 'date_utc': datetime.now(timezone.utc).isoformat(),
                 'requete': suivant['cas'], 'modele': MODELE, 'moteur': 'transformers-zerogpu',
                 'mode_saisie': mode}
    debut = perf_counter()
    question = prochaine_question(cas) if mode == 'guide' else None
    if question:
        reponse = {'statut': 'question', 'question': question}
    else:
        try:
            generation = generer(presentation(cas), cas.langue)
            reponse = interpreter(generation['texte'], generation['fin'])
            reponse.update({k: generation[k] for k in ('usage', 'prompt_sha256')})
        except Exception as erreur:
            # Je n'expose pas le message du moteur : il peut contenir des chemins ou secrets.
            reponse = {'statut': 'revue_humaine', 'esi': None, 'priorite': None,
                       'motif': 'Inférence indisponible ; aucune priorité attribuée.',
                       'erreur_type': type(erreur).__name__, 'validation_humaine_requise': True}
        suivant['termine'] = True
    evenement.update(reponse=reponse, duree_s=round(perf_counter() - debut, 3))
    suivant['evenements'].append(evenement)
    return suivant


def exporter_trace(etat):
    """Sérialise les seuls échanges de cette session, sans journal partagé."""
    return json.dumps({'session_id': etat['cas']['session_id'],
                       'usage': 'cas fictifs uniquement — prototype pédagogique',
                       'evenements': etat['evenements']}, ensure_ascii=False, indent=2)
