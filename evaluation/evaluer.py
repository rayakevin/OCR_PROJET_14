"""Je recalcule les métriques à partir de réponses de modèle déjà enregistrées."""

import argparse
import hashlib
import json
from pathlib import Path

from metrics import rapport_qa, rapport_triage


RACINE = Path(__file__).resolve().parents[1]
DONNEES = RACINE / "data/processed/sft_deux_phases"


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def main():
    """Évalue un fichier de prédictions associé à une phase et un split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--phase", choices=("qa", "triage"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=DONNEES)
    parser.add_argument("--limit", type=int, help="Évalue seulement les N premiers exemples du split.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prefixe = "phase1_qa" if args.phase == "qa" else "phase2_triage"
    fichier_donnees = args.data_dir / f"{prefixe}_{args.split}.jsonl"
    exemples = lire_jsonl(fichier_donnees)
    if args.limit is not None:
        exemples = exemples[: args.limit]
    predictions = lire_jsonl(args.predictions)
    rapport = (
        rapport_qa(exemples, predictions)
        if args.phase == "qa"
        else rapport_triage(exemples, predictions)
    )
    rapport |= {
        "phase": args.phase,
        "split": args.split,
        "fichier_predictions": str(args.predictions),
        "limite": args.limit,
        "donnees_sha256": hashlib.sha256(fichier_donnees.read_bytes()).hexdigest(),
    }
    sortie = args.output or args.predictions.with_name(args.predictions.stem + "_metriques.json")
    sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rapport, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
