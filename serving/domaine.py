"""Je sépare la collecte d'informations de l'interprétation de la sortie du modèle."""

import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Texte = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Champ = Literal['debut', 'evolution', 'contexte', 'retentissement']
QUESTIONS = {
    'plainte': ('Quel est le motif de consultation et quels symptômes sont décrits ?',
                'What is the reason for consultation and which symptoms are described?'),
    'debut': ('Depuis quand les symptômes sont-ils présents ?', 'When did the symptoms start?'),
    'evolution': ('Comment les symptômes évoluent-ils ?', 'How have the symptoms changed?'),
    'contexte': ('Quels âge, antécédents ou traitements sont connus ? Indiquer les informations inconnues.',
                 'What age, medical history or treatments are known? State what is unknown.'),
    'retentissement': ('Quel est le retentissement de la douleur sur les activités ?',
                       'How does the pain affect activities?'),
}


class Consultation(BaseModel):
    """Reçoit un cas fictif et les réponses déjà collectées, sans identité de patient."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    session_id: UUID | None = None
    langue: Literal['fr', 'en'] = 'fr'
    plainte: str = Field(default='', max_length=6000)
    reponses: dict[Champ, Texte] = Field(default_factory=dict)

    @model_validator(mode='after')
    def refuser_balises(self):
        """Évite l'injection directe de séparateurs du template ; ce n'est pas un filtre sémantique."""
        if any('<|' in v or '|>' in v for v in (self.plainte, *self.reponses.values())):
            raise ValueError('Les séparateurs de modèle ne sont pas acceptés.')
        return self


def prochaine_question(cas):
    """Demande uniquement un champ absent ; la branche douleur ne classe pas l'urgence."""
    champs = ['debut', 'evolution', 'contexte']
    if re.search(r'\b(douleur\w*|douloureu\w*|pain\w*)\b', cas.plainte, re.I):
        champs.insert(2, 'retentissement')
    manquant = 'plainte' if not cas.plainte else next((c for c in champs if c not in cas.reponses), None)
    return ({'champ': manquant, 'texte': QUESTIONS[manquant][cas.langue == 'en']}
            if manquant else None)


def presentation(cas):
    """Assemble uniquement les informations fournies, dans un ordre stable."""
    labels = {'debut': ('Début', 'Onset'), 'evolution': ('Évolution', 'Progression'),
              'contexte': ('Contexte', 'Background'), 'retentissement': ('Retentissement', 'Impact')}
    return '\n'.join([cas.plainte] + [f'{labels[c][cas.langue == "en"]} : {cas.reponses[c]}'
                                    for c in labels if c in cas.reponses])


def interpreter(texte, fin):
    """Exige une décision ESI/priorité unique et cohérente ; sinon s'abstient."""
    # Je ne répare pas une réponse ambiguë en inventant une priorité.
    esi = re.findall(r'^\s*(?:Niveau ESI|ESI level)\s*:\s*([1-5])\s*$', texte, re.M | re.I)
    priorites = re.findall(r'^\s*(?:Priorité|Priority)\s*:\s*(.+?)\s*$', texte, re.M | re.I)
    correspondance = {'urgence maximale': 'maximale', 'maximum urgency': 'maximale',
                     'urgence modérée': 'moderee', 'moderate urgency': 'moderee',
                     'urgence différée': 'differee', 'delayed urgency': 'differee'}
    priorite = correspondance.get(priorites[0].casefold()) if len(priorites) == 1 else None
    valeur = int(esi[0]) if len(esi) == 1 else None
    attendue = 'maximale' if valeur in (1, 2) else 'moderee' if valeur == 3 else 'differee'
    orientation = re.search(r'^\s*(?:Orientation|Conseils|Guidance)\s*:\s*\S.+$', texte, re.M | re.I)
    valide = fin == 'stop' and valeur is not None and priorite == attendue and bool(orientation)
    return {'statut': 'proposition_a_valider' if valide else 'revue_humaine',
            'esi': valeur if valide else None, 'priorite': priorite if valide else None,
            'motif': None if valide else 'Sortie incomplète, ambiguë ou incohérente.',
            'texte_modele': texte, 'validation_humaine_requise': True}
