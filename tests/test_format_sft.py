"""Je vérifie les frontières prompt/réponse et l'arrêt de génération."""

import unittest

from training.format_sft import construire_prompt, encoder_exemple
from training.entrainer_lora import CollateurSFT
from evaluation.inference import analyser_arret


class TokenizerTest:
    eos_token_id = 0
    chat_template = True

    def apply_chat_template(self, messages, **options):
        assert options["enable_thinking"] is False
        return "system/user/assistant/<think></think>/"

    def encode(self, texte, **options):
        return [ord(c) for c in texte]


class FormatSFTTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerTest()
        self.exemple = {"id": "test", "messages": [
            {"role": "system", "content": "Consigne"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Réponse complète."},
        ]}

    def test_prompt_identique_a_inference_et_masque(self):
        ligne = encoder_exemple(self.tokenizer, self.exemple, 100)
        prompt = self.tokenizer.encode(construire_prompt(self.tokenizer, self.exemple["messages"][:2]))
        self.assertEqual(ligne["input_ids"][:len(prompt)], prompt)
        self.assertEqual(ligne["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(ligne["labels"][len(prompt):], self.tokenizer.encode("Réponse complète.") + [0])

    def test_limite_exacte_conserve_eos_et_depassement_refuse(self):
        ligne = encoder_exemple(self.tokenizer, self.exemple, 100)
        self.assertEqual(encoder_exemple(self.tokenizer, self.exemple, ligne["length"]), ligne)
        with self.assertRaisesRegex(ValueError, "aucune troncature silencieuse"):
            encoder_exemple(self.tokenizer, self.exemple, ligne["length"] - 1)

    def test_padding_ne_masque_pas_le_vrai_eos(self):
        ligne = encoder_exemple(self.tokenizer, self.exemple, 100)
        lot = CollateurSFT(0)([ligne])
        fin = ligne["length"] - 1
        self.assertEqual(lot["labels"][0, fin].item(), 0)
        self.assertEqual(lot["attention_mask"][0, fin].item(), 1)
        self.assertTrue((lot["labels"][0, fin+1:] == -100).all())

    def test_eos_reel_ignore_padding_et_distingue_limite(self):
        self.assertEqual(analyser_arret([8, 7, 0, 0], [0]),
                         {"tokens_sortie": 3, "arret": "eos", "token_fin": 0})
        self.assertEqual(analyser_arret([8, 7, 9], [0])["arret"], "limite")


if __name__ == "__main__":
    unittest.main()
