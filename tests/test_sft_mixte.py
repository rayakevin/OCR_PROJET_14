"""Je vérifie le mélange sans modèle, GPU ni fichier de test final."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from training.preparer_mixte import construire_mixte


class TestMelangeSFT(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.racine = Path(self.tmp.name)
        self.donnees = self.racine / "sources"
        self.donnees.mkdir()
        for split in ("train", "validation"):
            for phase in ("phase1_qa", "phase2_triage"):
                nom = f"{phase}_{split}"
                lignes = [{"id": f"{nom}-{i}", "id_paire": f"groupe-{nom}-{i}",
                           "langue": langue, "messages": [
                               {"role": "system", "content": "Consigne"},
                               {"role": "user", "content": f"Question {nom} {i}"},
                               {"role": "assistant", "content": "Réponse"}]}
                          for i, langue in enumerate(("fr", "en"))]
                (self.donnees / f"{nom}.jsonl").write_text(
                    "".join(json.dumps(x) + "\n" for x in lignes))

    def test_union_reproductible_et_sources_preservees(self):
        avant = {p.name: p.read_bytes() for p in self.donnees.iterdir()}
        manifeste = construire_mixte(self.donnees, self.racine / "a")
        construire_mixte(self.donnees, self.racine / "b")
        self.assertEqual(manifeste["splits"]["train"]["par_tache"], {"qa": 2, "triage": 2})
        self.assertEqual(manifeste["splits"]["validation"]["par_langue"], {"fr": 2, "en": 2})
        for p in (self.racine / "a").iterdir():
            self.assertEqual(p.read_bytes(), (self.racine / "b" / p.name).read_bytes())
        self.assertEqual(avant, {p.name: p.read_bytes() for p in self.donnees.iterdir()})

    def test_refuse_fuite_de_groupe_ou_prompt(self):
        fichier = self.donnees / "phase1_qa_validation.jsonl"
        initial = fichier.read_text()
        for champ in ("id_paire", "messages"):
            with self.subTest(champ=champ):
                lignes = [json.loads(x) for x in initial.splitlines()]
                if champ == "id_paire":
                    lignes[0][champ] = "groupe-phase1_qa_train-0"
                else:
                    lignes[0][champ][1]["content"] = "  QUESTION   phase1_qa_train 0 "
                fichier.write_text("".join(json.dumps(x) + "\n" for x in lignes))
                with self.assertRaisesRegex(ValueError, "Chevauchement"):
                    construire_mixte(self.donnees, self.racine / "rejet")
                self.assertFalse((self.racine / "rejet").exists())
        fichier.write_text(initial)

    def test_refuse_doublon_et_ecrasement(self):
        sortie = self.racine / "sortie"
        construire_mixte(self.donnees, sortie)
        with self.assertRaisesRegex(ValueError, "vide"):
            construire_mixte(self.donnees, sortie)
        fichier = self.donnees / "phase1_qa_train.jsonl"
        fichier.write_text(fichier.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "dupliqués"):
            construire_mixte(self.donnees, self.racine / "doublon")


class TestDepartMixte(unittest.TestCase):
    def test_continuation_verifie_poids_corpus_et_epoque(self):
        import hashlib
        from training.continuer_mixte import verifier_depart
        from training.format_sft import FORMAT_SFT
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints/checkpoint-750"
            checkpoint.mkdir(parents=True)
            train, val = root / "train.jsonl", root / "validation.jsonl"
            train.write_text("train")
            val.write_text("validation")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"poids-source")
            etat = checkpoint / "trainer_state.json"
            etat.write_text(json.dumps({"epoch": 3, "global_step": 750}))
            protocole = {"arguments": {"phase": "sft_mixte", "revision": "figee"},
                         "format_sft": FORMAT_SFT,
                         "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                         "validation_sha256": hashlib.sha256(val.read_bytes()).hexdigest()}
            (root / "protocole_entrainement.json").write_text(json.dumps(protocole))
            depart = verifier_depart(checkpoint, train, val, "figee")
            self.assertEqual(depart["epoch_initiale"], 3)
            self.assertEqual(depart["optimiseur"], "reinitialise")
            with self.assertRaisesRegex(ValueError, "incompatibles"):
                verifier_depart(checkpoint, train, val, "autre")
            etat.write_text(json.dumps({"epoch": 2.5, "global_step": 625}))
            with self.assertRaisesRegex(ValueError, "complète"):
                verifier_depart(checkpoint, train, val, "figee")
            etat.write_text(json.dumps({"epoch": 3, "global_step": 750}))
            train.write_text("train modifié")
            with self.assertRaisesRegex(ValueError, "corpus train"):
                verifier_depart(checkpoint, train, val, "figee")
