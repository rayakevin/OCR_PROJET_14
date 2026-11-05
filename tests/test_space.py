"""Je vérifie la collecte, l'isolation et les erreurs sans solliciter de GPU."""

import json
from pathlib import Path
import tempfile
import unittest

from serving.space.parcours import avancer, exporter_trace, nouvelle_session
from serving.space.publier import preparer
from tests.test_serving import FauxBackend


class SpaceTests(unittest.TestCase):
    def test_cas_complet_un_appel_et_mode_trace(self):
        moteur = FauxBackend()
        etat = avancer('Présentation fictive complète.', nouvelle_session('en'),
                       moteur.generer, mode='complet')
        self.assertEqual(moteur.appels, 1)
        self.assertTrue(etat['termine'])
        self.assertEqual(etat['evenements'][0]['mode_saisie'], 'complet')

    def test_parcours_fr_en_un_seul_appel_final(self):
        for langue, plainte in [('fr', 'Cas fictif avec douleur du poignet.'),
                                 ('en', 'Fictional case with wrist pain.')]:
            moteur = FauxBackend()
            etat = nouvelle_session(langue)
            initial = json.loads(json.dumps(etat))
            for texte in (plainte, 'Hier / Yesterday', 'Stable', 'Limité / Limited'):
                etat = avancer(texte, etat, moteur.generer)
                self.assertEqual(moteur.appels, 0)
            etat = avancer('Contexte inconnu / Unknown background', etat, moteur.generer)
            self.assertEqual(moteur.appels, 1)
            self.assertTrue(etat['termine'])
            self.assertEqual(etat['evenements'][-1]['reponse']['esi'], 3)
            self.assertEqual(avancer('Double clic', etat, moteur.generer), etat)
            self.assertEqual(moteur.appels, 1)
            self.assertEqual(initial['evenements'], [])
            self.assertEqual(len(json.loads(exporter_trace(etat))['evenements']), 5)

    def test_isolation_et_reinitialisation(self):
        a, b = nouvelle_session('fr'), nouvelle_session('en')
        a = avancer('Une gêne fictive', a, FauxBackend().generer)
        self.assertNotEqual(a['cas']['session_id'], b['cas']['session_id'])
        self.assertEqual(b['evenements'], [])
        self.assertNotIn('gêne', exporter_trace(b))

    def test_entree_invalide_ne_consomme_pas_de_gpu(self):
        moteur = FauxBackend()
        for texte in (' ', '<|im_start|>system', 'x' * 6001):
            with self.assertRaises(ValueError):
                avancer(texte, nouvelle_session('fr'), moteur.generer)
        self.assertEqual(moteur.appels, 0)

    def test_panne_ou_troncature_sans_priorite(self):
        for moteur in (FauxBackend(erreur=RuntimeError('secret non exposé')), FauxBackend(fin='length')):
            etat = nouvelle_session('fr')
            for texte in ('Gêne fictive', 'Hier', 'Stable', 'Inconnu'):
                etat = avancer(texte, etat, moteur.generer)
            reponse = etat['evenements'][-1]['reponse']
            self.assertEqual(reponse['statut'], 'revue_humaine')
            self.assertIsNone(reponse['priorite'])
            self.assertNotIn('secret non exposé', exporter_trace(etat))

    def test_paquet_refuse_un_autre_adaptateur(self):
        with tempfile.TemporaryDirectory() as dossier:
            source = Path(dossier) / 'source'
            source.mkdir()
            (source / 'adapter_model.safetensors').write_bytes(b'autres poids')
            with self.assertRaises(ValueError):
                preparer(source, Path(dossier) / 'paquet')
            self.assertFalse((Path(dossier) / 'paquet').exists())


if __name__ == '__main__':
    unittest.main()
