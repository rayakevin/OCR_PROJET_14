import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from metrics import (  # noqa: E402
    extraire_esi,
    extraire_lettre_qcm,
    extraire_priorite,
    rapport_triage,
)


class ExtractionTests(unittest.TestCase):
    def test_extrait_esi_francais_et_anglais(self):
        self.assertEqual(extraire_esi("Niveau ESI : 2"), 2)
        self.assertEqual(extraire_esi("ESI level: 4"), 4)
        self.assertEqual(extraire_esi("**Niveau ESI :** 3"), 3)

    def test_priorite_peut_etre_deduite_de_esi(self):
        self.assertEqual(extraire_priorite("ESI level: 3"), "moderee")
        self.assertEqual(extraire_priorite("Priorité : urgence différée"), "differee")

    def test_extrait_une_lettre_annoncee(self):
        self.assertEqual(extraire_lettre_qcm("Réponse : C) proposition"), "C")
        self.assertIsNone(extraire_lettre_qcm("Je ne sais pas"))


class RapportTriageTests(unittest.TestCase):
    def test_avertissement_singulier_pluriel_et_absent(self):
        textes = [
            "Cet avis ne remplace pas une évaluation médicale.",
            "Ces conseils ne remplacent pas l'évaluation par un professionnel.",
            "This advice does not replace a medical evaluation.",
            "These recommendations do not replace a medical evaluation.",
            "Une évaluation médicale est nécessaire.",
        ]
        exemples = [{"id": str(i), "esi": 3, "langue": "fr"} for i in range(5)]
        predictions = [{"id": str(i), "prediction": "ESI : 3\n" + t}
                       for i, t in enumerate(textes)]
        self.assertEqual(rapport_triage(exemples, predictions)["taux_avertissement"], 0.8)

    def test_mesure_sous_triage_et_sortie_invalide(self):
        exemples = [
            {"id": "a", "esi": 2, "langue": "fr"},
            {"id": "b", "esi": 4, "langue": "fr"},
        ]
        predictions = [
            {"id": "a", "prediction": "Niveau ESI : 4"},
            {"id": "b", "prediction": "Réponse sans niveau"},
        ]
        rapport = rapport_triage(exemples, predictions)
        self.assertEqual(rapport["taux_sortie_esi_valide"], 0.5)
        self.assertEqual(rapport["taux_sous_triage_critique"], 1.0)

    def test_mesure_coherence_dune_paire_bilingue(self):
        exemples = [
            {"id": "a-fr", "id_paire": "a", "esi": 3, "langue": "fr"},
            {"id": "a-en", "id_paire": "a", "esi": 3, "langue": "en"},
        ]
        predictions = [
            {"id": "a-fr", "prediction": "Niveau ESI : 3"},
            {"id": "a-en", "prediction": "ESI level: 3"},
        ]
        rapport = rapport_triage(exemples, predictions)
        self.assertEqual(rapport["coherence_bilingue_esi"]["taux_meme_prediction"], 1.0)


if __name__ == "__main__":
    unittest.main()
