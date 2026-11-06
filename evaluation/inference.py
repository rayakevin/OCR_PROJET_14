"""Je génère des réponses déterministes pour comparer les checkpoints du projet."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from training.format_sft import construire_prompt
DONNEES = RACINE / "data/processed/sft_deux_phases"
MODELE_BASE = "Qwen/Qwen3-1.7B-Base"


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def analyser_arret(ids, fins):
    """Distingue un EOS produit par le modèle d'une coupure à la limite de tokens."""
    for position, identifiant in enumerate(ids):
        if identifiant in fins:
            return {"tokens_sortie": position + 1, "arret": "eos", "token_fin": identifiant}
    return {"tokens_sortie": len(ids), "arret": "limite", "token_fin": None}


def verifier_reprise(predictions, exemples, configuration, taille_lot):
    """Vérifie le préfixe conservé avant d'ajouter des réponses après une interruption."""
    ids = [ligne["id"] for ligne in predictions]
    if len(ids) > len(exemples) or ids != [x["id"] for x in exemples[:len(ids)]] or len(set(ids)) != len(ids):
        raise ValueError("Les identifiants conservés ne correspondent pas au début du corpus.")
    # Les premières sorties locales étaient exclusivement en bfloat16.
    if any(ligne.get(cle, "bfloat16" if cle == "dtype" else None) != valeur
           for ligne in predictions for cle, valeur in configuration.items()):
        raise ValueError("La configuration diffère de celle des réponses conservées.")
    if len(ids) < len(exemples) and len(ids) % taille_lot:
        raise ValueError("Le dernier lot est incomplet ; vérifier le fichier avant de reprendre.")
    return len(ids)


def main():
    """Exécute une inférence batchée et conserve les paramètres avec chaque réponse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qa", "triage"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test",
                        help="train sert à miner les préférences DPO, jamais à mesurer la généralisation.")
    parser.add_argument("--data-dir", type=Path, default=DONNEES, help="Version figée des partitions à évaluer.")
    parser.add_argument("--model", default=MODELE_BASE)
    parser.add_argument("--revision", help="Révision Hugging Face du socle pour figer les poids.")
    parser.add_argument("--adapter", type=Path, help="Adaptateur LoRA à charger sur le modèle de base.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Conserve les lots complets déjà produits.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16",
                        help="float16 permet l'inférence sur un GPU sans bfloat16 natif.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--stop-token", choices=("historique", "eos"), default="historique",
                        help="eos : arrêt natif, pour le SFT corrigé ; historique : EOS ou im_end.")
    args = parser.parse_args()

    prefixe = "phase1_qa" if args.phase == "qa" else "phase2_triage"
    fichier_donnees = args.data_dir / f"{prefixe}_{args.split}.jsonl"
    exemples = lire_jsonl(fichier_donnees)
    if args.limit:
        exemples = exemples[:args.limit]
    configuration = {
        "modele": args.model,
        "revision": args.revision,
        "adaptateur": str(args.adapter.resolve()) if args.adapter else None,
        "phase": args.phase,
        "split": args.split,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "dtype": args.dtype,
    }
    if args.stop_token != "historique":
        configuration["generation"]["stop_token"] = args.stop_token
    if args.data_dir.resolve() != DONNEES.resolve():
        configuration["donnees_sha256"] = hashlib.sha256(fichier_donnees.read_bytes()).hexdigest()
    if args.split == "train":
        if args.adapter is None:
            raise ValueError("Le minage attend un adaptateur SFT identifié.")
        configuration["adaptateur_sha256"] = hashlib.sha256(
            (args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
    position_depart = 0
    if args.resume and args.output.exists():
        position_depart = verifier_reprise(
            lire_jsonl(args.output), exemples, configuration, args.batch_size
        )
        if args.output.stat().st_size:
            with args.output.open("rb") as fichier:
                fichier.seek(-1, os.SEEK_END)
                if fichier.read(1) != b"\n":
                    raise ValueError("La dernière ligne n'est pas terminée ; vérifier le JSONL.")
        print(f"Reprise : {position_depart}/{len(exemples)} réponses conservées.", flush=True)
        if position_depart == len(exemples):
            return
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "left"
    if args.split == "train":
        # Un cas amputé ne doit pas devenir une préférence d'entraînement.
        for exemple in exemples:
            prompt = construire_prompt(tokenizer, exemple['messages'][:2])
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > 2048:
                raise ValueError(f"Minage sans troncature impossible : {exemple['id']}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    modele = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=getattr(torch, args.dtype),
        device_map="auto",
    )
    if args.adapter is not None:
        # Je conserve le tokenizer du socle : seul l'adaptateur change dans la comparaison.
        modele = PeftModel.from_pretrained(modele, str(args.adapter.resolve()), is_trainable=False)
    modele.eval()
    fin_message = tokenizer.convert_tokens_to_ids("<|im_end|>")
    fins = [tokenizer.eos_token_id]
    if args.stop_token == "historique" and isinstance(fin_message, int) and fin_message >= 0:
        fins.append(fin_message)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    debut = time.perf_counter()
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as fichier, torch.inference_mode():
        for position in range(position_depart, len(exemples), args.batch_size):
            lot = exemples[position:position + args.batch_size]
            prompts = [construire_prompt(tokenizer, x["messages"][:2]) for x in lot]
            tokens = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(modele.device)
            sorties = modele.generate(
                **tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=fins,
            )
            longueurs = tokens["attention_mask"].sum(dim=1).tolist()
            for exemple, sortie, longueur in zip(lot, sorties, longueurs):
                # Avec le padding gauche, la génération commence après la largeur commune du batch.
                ids_sortie = sortie[tokens["input_ids"].shape[1]:].tolist()
                arret = analyser_arret(ids_sortie, fins)
                prediction = tokenizer.decode(ids_sortie[:arret["tokens_sortie"]], skip_special_tokens=True).strip()
                fichier.write(json.dumps({
                    **configuration,
                    "id": exemple["id"],
                    "prediction": prediction,
                    "modele": args.model,
                    "revision": args.revision,
                    "adaptateur": str(args.adapter.resolve()) if args.adapter else None,
                    "dtype": args.dtype,
                    "phase": args.phase,
                    "split": args.split,
                    "generation": configuration["generation"],
                    **arret,
                    "date_utc": datetime.now(timezone.utc).isoformat(),
                    "tokens_entree": int(longueur),
                }, ensure_ascii=False) + "\n")
            fichier.flush()
            os.fsync(fichier.fileno())
            print(f"{min(position + len(lot), len(exemples))}/{len(exemples)}", flush=True)
    print(json.dumps({
        "exemples": len(exemples),
        "secondes": round(time.perf_counter() - debut, 2),
        "sortie": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
