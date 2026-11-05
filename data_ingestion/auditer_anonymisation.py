"""Je repère les identifiants potentiels dans des fichiers JSONL sans modifier les textes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider


RACINE = Path(__file__).resolve().parents[1]
DOSSIER = RACINE / "data/processed/sft_deux_phases"
ENTITES = [
    "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS", "URL",
    "LOCATION", "DATE_TIME", "MEDICAL_LICENSE", "FR_RPPS",
]


def lire_jsonl(chemin):
    """Charge les exemples d’un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def moteur():
    """Configure Presidio et les modèles français et anglais de spaCy."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }
    nlp = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    analyseur = AnalyzerEngine(nlp_engine=nlp, supported_languages=["fr", "en"])
    analyseur.registry.add_recognizer(PatternRecognizer(
        supported_entity="FR_RPPS",
        patterns=[Pattern("RPPS_11_CHIFFRES", r"(?<!\d)\d{11}(?!\d)", 0.7)],
        supported_language="fr",
    ))
    return analyseur


def auditer(fichiers):
    """Analyse les messages des fichiers fournis et écrit un rapport sans recopier les extraits."""
    analyseur = moteur()
    lignes = []
    exemples_analyses = 0
    messages_analyses = 0
    for chemin in fichiers:
        for exemple in lire_jsonl(chemin):
            exemples_analyses += 1
            langue = exemple["langue"]
            for message in exemple["messages"]:
                if message["role"] == "system":
                    continue
                messages_analyses += 1
                texte = message["content"]
                for detection in analyseur.analyze(
                    text=texte, language=langue, entities=ENTITES, score_threshold=0.65
                ):
                    extrait = texte[detection.start:detection.end]
                    lignes.append({
                        "id": exemple["id"],
                        "fichier": chemin.name,
                        "role": message["role"],
                        "langue": langue,
                        "entite": detection.entity_type,
                        "score": round(detection.score, 3),
                        "debut": detection.start,
                        "fin": detection.end,
                        "sha256_extrait": hashlib.sha256(extrait.casefold().encode()).hexdigest(),
                    })

    compte_type = Counter(x["entite"] for x in lignes)
    compte_exemple = Counter(x["id"] for x in lignes)
    rapport = {
        "exemples_analyses": exemples_analyses,
        "messages_analyses": messages_analyses,
        "seuil": 0.65,
        "detections": len(lignes),
        "exemples_avec_detection": len(compte_exemple),
        "par_entite": {entite: compte_type.get(entite, 0) for entite in ENTITES},
        "identifiants_directs_detectes": sum(
            compte_type.get(entite, 0)
            for entite in ("PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS", "MEDICAL_LICENSE", "FR_RPPS")
        ),
        "methode": "Presidio 2.2.364, spaCy fr_core_news_md 3.8.0 et en_core_web_sm 3.8.0",
        "statut": "audit_automatique_termine_revue_humaine_requise",
        "limite": "Une détection automatique peut être un faux positif médical.",
    }
    (DOSSIER / "audit_anonymisation.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes)
    )
    (DOSSIER / "rapport_anonymisation.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return rapport


if __name__ == "__main__":
    parseur = argparse.ArgumentParser()
    parseur.add_argument(
        "fichiers",
        nargs="*",
        type=Path,
        default=[
            DOSSIER / "phase1_qa_train.jsonl",
            DOSSIER / "phase1_qa_validation.jsonl",
            DOSSIER / "phase1_qa_test.jsonl",
            DOSSIER / "phase2_triage_train.jsonl",
            DOSSIER / "phase2_triage_validation.jsonl",
            DOSSIER / "phase2_triage_test.jsonl",
        ],
    )
    auditer(parseur.parse_args().fichiers)
