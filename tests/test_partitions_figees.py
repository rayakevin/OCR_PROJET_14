"""Je contrôle les fuites de prompts, même quand les identifiants diffèrent."""

import unittest
from data_ingestion.figer_partitions import nettoyer, verifier


def cas(identifiant, fr, en):
    return [{'id': f'{identifiant}-{langue}', 'id_paire': identifiant, 'langue': langue,
             'messages': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': texte}]}
            for langue, texte in (('fr', fr), ('en', en))]


class PartitionsTests(unittest.TestCase):
    def sources(self):
        return {f'{p}_{s}': [] for p in ('phase1_qa', 'phase2_triage', 'dpo')
                for s in ('train', 'validation', 'test') if p != 'dpo' or s != 'test'}

    def test_doublon_anglais_exclut_les_deux_langues_du_train(self):
        s = self.sources()
        s['phase2_triage_train'] = cas('a', 'traduction A', ' Same  question ')
        s['phase2_triage_validation'] = cas('b', 'traduction B', 'same question')
        sorties, exclusions = nettoyer(s, set())
        self.assertEqual(sorties['phase2_triage_train'], [])
        self.assertEqual(len(exclusions[0]['ids']), 2)
        verifier(sorties)

    def test_cas_expose_exclu_du_test_et_jamais_recycle_en_train(self):
        s = self.sources()
        s['phase2_triage_test'] = cas('a', 'cas FR', 'case EN')
        s['phase2_triage_train'] = cas('b', 'autre traduction', 'case EN')
        sorties, _ = nettoyer(s, {'a-fr'})
        self.assertFalse(sorties['phase2_triage_test'])
        self.assertFalse(sorties['phase2_triage_train'])

    def test_dpo_exclut_prompt_sft_et_priorise_validation(self):
        s = self.sources()
        s['phase2_triage_validation'] = cas('a', 'cas FR', 'case EN')
        def paire(i, q):
            return {'id': i, 'prompt': [{'content': q}],
                    'chosen': [{'content': 'oui'}], 'rejected': [{'content': 'non'}]}
        s['dpo_validation'] = [paire('v', 'question')]
        s['dpo_train'] = [paire('t', 'QUESTION'), paire('fuite', 'CASE EN'), paire('ok', 'distinct')]
        sorties, _ = nettoyer(s, set())
        self.assertEqual([r['id'] for r in sorties['dpo_train']], ['ok'])
        verifier(sorties)


if __name__ == '__main__':
    unittest.main()
