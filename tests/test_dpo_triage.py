"""Je vérifie l'absence de fuite et les garde-fous, sans modèle ni GPU."""

import unittest
from training.preparer_dpo_triage import miner, repartir, decision
from evaluation.comparer_dpo_triage import comparer


def texte(esi):
    priorite = 'maximale' if esi <= 2 else 'modérée' if esi == 3 else 'différée'
    return f'Niveau ESI : {esi}\nPriorité : urgence {priorite}\nOrientation : Évaluation médicale.'


class DPOCibleTests(unittest.TestCase):
    def setUp(self):
        self.exemple = {'id': 'a', 'id_paire': 'cas-a', 'langue': 'fr', 'esi': 2, 'urgence': 'maximale',
            'messages': [{'role': 'system', 'content': 'Consigne'}, {'role': 'user', 'content': 'Cas fictif'},
                         {'role': 'assistant', 'content': texte(2)}]}
        self.prediction = {'id': 'a', 'split': 'train', 'phase': 'triage', 'revision': 'rev',
            'donnees_sha256': 'source', 'adaptateur_sha256': 'poids', 'arret': 'eos', 'prediction': texte(3)}

    def extraire(self, pred):
        return miner([self.exemple], [pred], 'rev', 'source', 'poids')

    def test_erreur_reelle_et_cible_inchangee(self):
        retenus, _ = self.extraire(self.prediction)
        self.assertEqual(retenus[0]['label_type'], 'sous_triage_critique')
        self.assertEqual(retenus[0]['chosen'], [self.exemple['messages'][2]])
        self.assertEqual(retenus[0]['rejected'][0]['content'], self.prediction['prediction'])

    def test_refuse_validation_et_mauvais_poids(self):
        for modification in ({'split': 'validation'}, {'split': 'test'}, {'adaptateur_sha256': 'autre'},
                             {'id': 'autre'}, {'donnees_sha256': 'autre'}):
            with self.assertRaises(ValueError): self.extraire(self.prediction | modification)

    def test_exclut_correct_et_tronque(self):
        for modification in ({'arret': 'limite'}, {'prediction': texte(2)}, {'prediction': 'invalide'}):
            retenus, exclusions = self.extraire(self.prediction | modification)
            self.assertFalse(retenus)
            self.assertEqual(sum(exclusions.values()), 1)

    def test_separation_des_groupes_bilingues(self):
        paires = [{'id': f'{i}-{langue}-{cat}', 'id_paire': str(i), 'langue': langue, 'label_type': cat}
                  for i in range(20) for langue in ('fr', 'en')
                  for cat in ('sur_triage', 'sous_triage_critique')]
        splits = repartir(paires)
        self.assertFalse({x['id_paire'] for x in splits['train']} &
                         {x['id_paire'] for x in splits['validation']})
        self.assertEqual(splits, repartir(paires))
        with self.assertRaises(ValueError):
            repartir([x for x in paires if x['label_type'] == 'sur_triage'])

    def test_variantes_du_corpus_sans_assouplir_api(self):
        self.assertEqual(decision('ESI Level: 2\nPriority: Maximum emergency'), 2)
        self.assertEqual(decision('niveau_ESI: 4\npriorité: urgence différée'), 4)
        self.assertIsNone(decision('ESI Level: 3\nPriority: Maximum emergency'))
        self.assertIsNone(decision(texte(2) + '\nNiveau ESI : 3'))

    def test_rappel_seul_ne_suffit_pas(self):
        ref = {'rappel_maximale': .91, 'macro_f1_priorite': .7874, 'taux_sur_triage': .1807,
               'accuracy_qcm': .5725, 'f1_lexical': .4341, 'esi_analysable': 1.}
        self.assertFalse(comparer(ref, ref)['signal_favorable'])
        self.assertTrue(comparer(ref, ref | {'rappel_maximale': .94})['signal_favorable'])
        self.assertFalse(comparer(ref, ref | {'rappel_maximale': 1., 'taux_sur_triage': .8})['signal_favorable'])


if __name__ == '__main__':
    unittest.main()
