"""Je compare le DPO ciblé au SFT mixte avec des garde-fous fixés avant le run."""


def scores(qa, triage):
    """Sépare les capacités QA, le classement et les erreurs de priorité."""
    return {'accuracy_qcm': qa['qcm']['accuracy'],
            'f1_lexical': qa['reponses_ouvertes']['f1_lexical_moyen'],
            'macro_f1_priorite': triage['priorite']['macro_f1'],
            'rappel_maximale': triage['priorite']['rappel_par_classe']['maximale'],
            'accuracy_esi': triage['esi']['accuracy'],
            'taux_sur_triage': triage['taux_sur_triage'],
            'esi_analysable': triage['taux_sortie_esi_valide'],
            'coherence_bilingue': triage['coherence_bilingue_esi']['taux_meme_prediction']}


def comparer(reference, candidat):
    """Un signal favorable exige plusieurs critères, pas seulement la loss DPO.

    Ces tolérances expérimentales ne sont pas des seuils de sécurité clinique.
    Même favorable, le résultat n'entraîne aucune promotion automatique.
    """
    conditions = {
        'rappel_urgences_en_hausse': candidat['rappel_maximale'] > reference['rappel_maximale'],
        'macro_f1_recul_max_1_point': candidat['macro_f1_priorite'] >= reference['macro_f1_priorite'] - .01,
        'sur_triage_hausse_max_2_points': candidat['taux_sur_triage'] is not None and
            candidat['taux_sur_triage'] <= reference['taux_sur_triage'] + .02,
        'qcm_recul_max_3_points': candidat['accuracy_qcm'] >= reference['accuracy_qcm'] - .03,
        'qa_ouverte_recul_max_3_points': candidat['f1_lexical'] >= reference['f1_lexical'] - .03,
        'format_sans_recul': candidat['esi_analysable'] >= reference['esi_analysable'],
    }
    return {'conditions': conditions, 'signal_favorable': all(conditions.values()),
            'promotion_automatique': False,
            'limite': 'Validation exploratoire ; cas bilingues corrélés ; confirmation indépendante nécessaire.'}
