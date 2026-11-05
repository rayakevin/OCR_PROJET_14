"""Je vérifie le format DPO et une vraie mise à jour sur un modèle minuscule CPU."""

import tempfile
import unittest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl import DPOConfig, DPOTrainer

from training.entrainer_dpo import empreinte_adaptateur, verifier_tokenisation
from training.preparer_dpo import formater_paire


class FormatDPOTests(unittest.TestCase):
    def tokenizer(self):
        moteur = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, '[PAD]': 2,
                                     'question': 3, 'oui': 4, 'non': 5}, unk_token='[UNK]'))
        moteur.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=moteur, unk_token='[UNK]',
                                      eos_token='[EOS]', pad_token='[PAD]', padding_side='left')

    def test_refuse_troncature_et_conserve_eos(self):
        t = self.tokenizer()
        r = {'id': 'a', 'label_type': 'hard', 'prompt': [{'role': 'user', 'content': 'question'}],
             'chosen': [{'role': 'assistant', 'content': 'oui'}],
             'rejected': [{'role': 'assistant', 'content': 'non'}]}
        paire, longueur = formater_paire(r, t, 1000, 'consigne')
        self.assertTrue(paire['chosen'].endswith(t.eos_token))
        self.assertIsNone(formater_paire(r, t, longueur - 1, 'consigne')[0])

    def test_trl_garde_reference_gelee_et_frontieres(self):
        torch.manual_seed(3407)
        torch.set_num_threads(2)
        t = self.tokenizer()
        config = Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, pad_token_id=2, eos_token_id=1, use_cache=False)
        modele = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        debut = empreinte_adaptateur(modele, 'default')
        lignes = [{'id': str(i), 'prompt': 'question ', 'chosen': 'oui[EOS]', 'rejected': 'non[EOS]'}
                  for i in range(4)]
        with tempfile.TemporaryDirectory() as dossier:
            args = DPOConfig(output_dir=dossier, max_steps=2, per_device_train_batch_size=2,
                per_device_eval_batch_size=2, learning_rate=1e-3, use_cpu=True, bf16=False,
                gradient_checkpointing=False, max_length=32, report_to='none', save_strategy='no',
                precompute_ref_log_probs=True, precompute_ref_batch_size=2, seed=3407)
            trainer = DPOTrainer(model=modele, args=args, processing_class=t,
                                 train_dataset=Dataset.from_list(lignes), eval_dataset=Dataset.from_list(lignes))
            self.assertEqual(empreinte_adaptateur(modele, 'ref'), debut)
            verifier_tokenisation(trainer.train_dataset, lignes, t, 32)
            trainer.evaluate()
            trainer.train()
            self.assertEqual(empreinte_adaptateur(modele, 'ref'), debut)
            self.assertNotEqual(empreinte_adaptateur(modele, 'default'), debut)


if __name__ == '__main__':
    unittest.main()
