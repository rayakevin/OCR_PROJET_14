"""Je distingue les erreurs de priorité, les traductions et les régressions."""
import unittest
from evaluation.revoir_urgences import analyser_erreurs


class TestRevueUrgences(unittest.TestCase):
    def test_compte_cas_sans_doubler_traductions(self):
        exemples = [{"id": langue, "id_paire": "cas", "langue": langue, "esi": 2,
                     "messages": [{"content": "consigne"}, {"content": "cas"}, {"content": "ESI 2"}]}
                    for langue in ("fr", "en")]
        pred = [{"id": langue, "prediction": "ESI 3"} for langue in ("fr", "en")]
        ref = [{"id": "fr", "prediction": "ESI 3"}, {"id": "en", "prediction": "ESI 2"}]
        rapport = analyser_erreurs(exemples, pred, ref, {"750": pred})
        self.assertEqual(rapport["nombre_entrees"], 2)
        self.assertEqual(rapport["nombre_cas"], 1)
        self.assertEqual(rapport["deja_manquees_par_reference"], 1)
        with self.assertRaisesRegex(ValueError, "incomplètes"):
            analyser_erreurs(exemples, pred[:-1], ref, {})

    def test_invalide_compte_comme_urgence_manquee(self):
        exemple = {"id": "x", "langue": "fr", "esi": 1,
                   "messages": [{"content": "consigne"}, {"content": "cas"}, {"content": "ESI 1"}]}
        pred = [{"id": "x", "prediction": "Je ne sais pas"}]
        rapport = analyser_erreurs([exemple], pred, pred, {})
        self.assertEqual(rapport["nombre_entrees"], 1)
        self.assertIsNone(rapport["erreurs"][0]["esi_predit"])
