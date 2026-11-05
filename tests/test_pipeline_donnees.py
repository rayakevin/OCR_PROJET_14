import copy
import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_ingestion"))

from consolider_consensus_trois import consensus_trois
from consolider_ultramedical import calculer_consensus
from outils_sft import dedupliquer, exemple
from schema_annotation import fingerprint, valider


class SchemaAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.cas = {
            "case_id": "cas-test",
            "texte_source": "Patient conscient avec une douleur thoracique.",
        }
        self.annotation = {
            "case_id": "cas-test",
            "presentation_patient": "oui",
            "temporalite": "initial",
            "decision_entree": "candidat_initial",
            "extraits_qualification": ["Patient conscient"],
            "informations_a_preciser": [],
            "justification_qualification": "Présentation clinique initiale.",
            "statut_classement": "classe",
            "esi": 3,
            "urgence": "moderee",
            "signes_alerte": [],
            "justification_classement": "Évaluation et ressources nécessaires.",
        }

    def test_annotation_valide(self):
        self.assertEqual(valider(self.annotation, self.cas), self.annotation)

    def test_citation_inventee_refusee(self):
        annotation = copy.deepcopy(self.annotation)
        annotation["extraits_qualification"] = ["Citation absente"]
        with self.assertRaises(ValueError):
            valider(annotation, self.cas)

    def test_esi_booleen_et_urgence_incoherente_refuses(self):
        for modification in ({"esi": True}, {"urgence": "maximale"}):
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                valider(self.annotation | modification, self.cas)

    def test_abstention_exige_des_valeurs_nulles(self):
        annotation = self.annotation | {
            "statut_classement": "insuffisant",
            "esi": None,
            "urgence": None,
        }
        self.assertEqual(valider(annotation, self.cas), annotation)
        with self.assertRaises(ValueError):
            valider(annotation | {"esi": 3}, self.cas)

    def test_empreinte_change_avec_le_contenu(self):
        self.assertNotEqual(fingerprint({"a": 1}), fingerprint({"a": 2}))


class ConsensusTests(unittest.TestCase):
    def annotation(self, decision="candidat_initial", urgence="moderee", esi=3):
        return {
            "decision_entree": decision,
            "statut_classement": "classe" if decision == "candidat_initial" else "non_classe",
            "urgence": urgence if decision == "candidat_initial" else None,
            "esi": esi if decision == "candidat_initial" else None,
        }

    def test_majorite_exacte_retenue(self):
        annotations = {
            "codex": self.annotation(),
            "claude": self.annotation(),
            "mistral": self.annotation(urgence="maximale", esi=2),
        }
        resultat = consensus_trois(annotations)
        self.assertEqual((resultat["statut"], resultat["esi"]), ("retenu", 3))

    def test_absence_de_majorite_ecartee(self):
        annotations = {
            "codex": self.annotation(),
            "claude": self.annotation(urgence="maximale", esi=2),
            "mistral": self.annotation(urgence="differee", esi=4),
        }
        self.assertEqual(consensus_trois(annotations)["statut"], "a_revoir")

    def test_ultramedical_exige_un_esi_majoritaire(self):
        annotations = [
            self.annotation(urgence="maximale", esi=1),
            self.annotation(urgence="maximale", esi=2),
            self.annotation(urgence="maximale", esi=2),
        ]
        resultat = calculer_consensus(annotations)
        self.assertEqual((resultat["statut"], resultat["esi"]), ("retenu", 2))

        annotations[2] = self.annotation(urgence="moderee", esi=3)
        self.assertEqual(calculer_consensus(annotations)["statut"], "a_revoir")


class MiseEnFormeTests(unittest.TestCase):
    def test_dedoublonnage_sur_le_message_utilisateur(self):
        premier = exemple("a", "s", "Même question", "Réponse 1", langue="fr", type_tache="test")
        doublon = exemple("b", "s", "  même   QUESTION ", "Réponse 2", langue="fr", type_tache="test")
        self.assertEqual(dedupliquer([premier, doublon]), [premier])


if __name__ == "__main__":
    unittest.main()
