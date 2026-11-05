"""Je vérifie qu'une reprise ne mélange pas des réponses incompatibles."""

import unittest

from evaluation.inference import verifier_reprise


class RepriseTests(unittest.TestCase):
    def setUp(self):
        self.exemples = [{"id": str(i)} for i in range(6)]
        self.configuration = {"modele": "socle", "adaptateur": "/adaptateur"}
        self.predictions = [dict(x, **self.configuration) for x in self.exemples]

    def test_reprend_apres_un_lot_complet(self):
        self.assertEqual(verifier_reprise(
            self.predictions[:4], self.exemples, self.configuration, 4
        ), 4)

    def test_accepte_un_corpus_termine_avec_un_dernier_lot_court(self):
        self.assertEqual(verifier_reprise(
            self.predictions, self.exemples, self.configuration, 4
        ), 6)

    def test_refuse_un_lot_incomplet_ou_des_identifiants_desordonnes(self):
        for lignes in (self.predictions[:3], self.predictions[:4][::-1]):
            with self.subTest(lignes=lignes), self.assertRaises(ValueError):
                verifier_reprise(lignes, self.exemples, self.configuration, 4)

    def test_refuse_un_autre_adaptateur(self):
        configuration = self.configuration | {"adaptateur": "/autre"}
        with self.assertRaises(ValueError):
            verifier_reprise(self.predictions[:4], self.exemples, configuration, 4)

    def test_refuse_de_melanger_les_precisions(self):
        configuration = self.configuration | {"dtype": "float16"}
        with self.assertRaises(ValueError):
            verifier_reprise(self.predictions[:4], self.exemples, configuration, 4)
        # Les sorties locales antérieures à ce champ étaient en bfloat16.
        self.assertEqual(verifier_reprise(
            self.predictions[:4], self.exemples,
            self.configuration | {"dtype": "bfloat16"}, 4
        ), 4)


if __name__ == "__main__":
    unittest.main()
