"""J'assemble les paires bilingues MediQAl et UltraMedical de la phase 2."""

import hashlib
import json
from pathlib import Path

from outils_sft import SYSTEME_FR


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/processed/sft_deux_phases"
SEED = 3407


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def ecrire_jsonl(chemin, lignes):
    """Écrit une collection d'objets au format JSONL."""
    chemin.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes))


def repartir_par_paire(exemples, proportion_validation=0.10):
    """Place les deux langues d'un cas dans le même split reproductible."""
    groupes = {}
    for exemple in exemples:
        groupes.setdefault(exemple["id_paire"], []).append(exemple)
    ids = sorted(
        groupes,
        key=lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest(),
    )
    validation = set(ids[:round(len(ids) * proportion_validation)])
    train = [x for identifiant, groupe in groupes.items() if identifiant not in validation for x in groupe]
    val = [x for identifiant, groupe in groupes.items() if identifiant in validation for x in groupe]
    return train, val


def assembler():
    """Valide les traductions UltraMedical puis assemble les quatre blocs."""
    ultra_en = lire_jsonl(BASE / "phase2_ultramedical_en.jsonl")
    traductions = {x["id"]: x for x in lire_jsonl(BASE / "traduction_ultra_sorties.jsonl")}
    if set(traductions) != {x["id"] for x in ultra_en}:
        raise ValueError("Traductions UltraMedical incomplètes")

    ultra_fr = []
    for source in ultra_en:
        traduction = traductions[source["id"]]
        ultra_fr.append(source | {
            "id": source["id"] + "-fr",
            "langue": "fr",
            "niveau_confiance": source["niveau_confiance"] + "+traduction_mistral",
            "messages": [
                {"role": "system", "content": SYSTEME_FR},
                {"role": "user", "content": traduction["texte_fr"]},
                {"role": "assistant", "content": traduction["reponse_fr"]},
            ],
        })
    ultra_train, ultra_val = repartir_par_paire(ultra_en + ultra_fr)
    train = lire_jsonl(BASE / "phase2_mediqal_train.jsonl") + ultra_train
    validation = lire_jsonl(BASE / "phase2_mediqal_validation.jsonl") + ultra_val
    ensemble = train + validation

    ids = [x["id"] for x in ensemble]
    if len(ids) != len(set(ids)):
        raise ValueError("Identifiants dupliqués")
    if any([m["role"] for m in x["messages"]] != ["system", "user", "assistant"] for x in ensemble):
        raise ValueError("Schéma de messages incorrect")
    if len(ensemble) != 1400 + 2 * len(ultra_en):
        raise ValueError("Volume final de la phase 2 incorrect")

    ecrire_jsonl(BASE / "phase2_triage_train.jsonl", train)
    ecrire_jsonl(BASE / "phase2_triage_validation.jsonl", validation)
    rapport = {
        "train": len(train),
        "validation": len(validation),
        "total": len(ensemble),
        "langues": {
            langue: sum(x["langue"] == langue for x in ensemble) for langue in ("fr", "en")
        },
        "sources": {
            source: sum(x["source"] == source for x in ensemble)
            for source in sorted({x["source"] for x in ensemble})
        },
    }
    (BASE / "rapport_phase2.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return rapport


if __name__ == "__main__":
    assembler()
