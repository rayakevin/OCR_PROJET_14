"""Je sépare le SFT médical général de la spécialisation au triage."""

import json
from collections import Counter
from pathlib import Path

from outils_sft import (
    RACINE,
    SEED,
    charger_medquad,
    charger_mediqal,
    charger_triage,
    dedupliquer,
    ecrire_jsonl,
    ordre_stable,
    statistiques,
)


SORTIE = RACINE / "data/processed/sft_deux_phases"
N_QA_LANGUE = 1311
N_QA_VALIDATION_LANGUE = 110
N_TRIAGE_VALIDATION_PAIRES = 70


def repartition(lignes, nombre_validation, suffixe):
    """Sépare une liste de façon déterministe en entraînement et validation."""
    lignes = sorted(lignes, key=lambda x: ordre_stable(x["id"] + suffixe))
    return lignes[nombre_validation:], lignes[:nombre_validation]


def preparer_phase_qa():
    """Construit un corpus QA équilibré entre français et anglais."""
    francais = charger_mediqal(N_QA_LANGUE)
    anglais = charger_medquad(N_QA_LANGUE)
    if len(francais) != N_QA_LANGUE or len(anglais) != N_QA_LANGUE:
        raise ValueError("Réservoir QA bilingue insuffisant")
    train_fr, validation_fr = repartition(francais, N_QA_VALIDATION_LANGUE, ":qa-fr")
    train_en, validation_en = repartition(anglais, N_QA_VALIDATION_LANGUE, ":qa-en")
    train = sorted(train_fr + train_en, key=lambda x: ordre_stable(x["id"] + ":qa-train"))
    validation = sorted(
        validation_fr + validation_en, key=lambda x: ordre_stable(x["id"] + ":qa-validation")
    )
    return train, validation


def preparer_triage_fr():
    """Charge et déduplique les 700 cas MediQAl retenus."""
    lignes = dedupliquer(charger_triage())
    if len(lignes) != 700:
        raise ValueError(f"700 cas de triage uniques attendus, {len(lignes)} obtenus")
    return sorted(lignes, key=lambda x: ordre_stable(x["id"] + ":paire-triage"))


def statistiques_phase(lignes):
    """Ajoute la distribution ESI aux statistiques générales."""
    base = statistiques(lignes)
    base["repartition_esi"] = dict(Counter(str(x["esi"]) for x in lignes if "esi" in x))
    return base


def preparer():
    """Produit les fichiers disponibles et le manifeste de la stratégie en deux phases."""
    SORTIE.mkdir(parents=True, exist_ok=True)
    qa_train, qa_validation = preparer_phase_qa()
    triage_fr = preparer_triage_fr()
    validation_ids = {x["id"] for x in triage_fr[:N_TRIAGE_VALIDATION_PAIRES]}
    triage_train_fr = [x for x in triage_fr if x["id"] not in validation_ids]
    triage_validation_fr = [x for x in triage_fr if x["id"] in validation_ids]

    ecrire_jsonl(SORTIE / "phase1_qa_train.jsonl", qa_train)
    ecrire_jsonl(SORTIE / "phase1_qa_validation.jsonl", qa_validation)
    ecrire_jsonl(SORTIE / "phase1_qa_corpus.jsonl", qa_train + qa_validation)
    ecrire_jsonl(SORTIE / "phase2_triage_fr_train.jsonl", triage_train_fr)
    ecrire_jsonl(SORTIE / "phase2_triage_fr_validation.jsonl", triage_validation_fr)

    demandes = []
    for ligne in triage_fr:
        demandes.append({
            "id": ligne["id"],
            "split_cible": "validation" if ligne["id"] in validation_ids else "train",
            "texte_fr": ligne["messages"][1]["content"],
            "reponse_fr": ligne["messages"][2]["content"],
            "esi": ligne["esi"],
            "urgence": ligne["urgence"],
        })
    ecrire_jsonl(SORTIE / "traduction_triage_entrees.jsonl", demandes)

    rapport = {
        "seed": SEED,
        "strategie": "SFT_sequentiel_meme_adaptateur_LoRA",
        "phase1_qa": {
            "train": statistiques_phase(qa_train),
            "validation": statistiques_phase(qa_validation),
        },
        "phase2_triage_disponible_fr": {
            "train": statistiques_phase(triage_train_fr),
            "validation": statistiques_phase(triage_validation_fr),
        },
        "phase2_triage_mediqal_attendu_apres_traduction": {
            "train": 1260,
            "validation": 140,
            "langues": {"fr": 700, "en": 700},
        },
        "phase2_triage_ultramedical_attendu": {
            "nombre": 978,
            "langues": {"en": 489, "fr": 489},
            "statut": "consensus_et_traductions_valides",
        },
        "total_attendu": 5000,
        "statut": "deux_phases_preparees_a_repartir",
    }
    (SORTIE / "rapport.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return rapport


if __name__ == "__main__":
    preparer()
