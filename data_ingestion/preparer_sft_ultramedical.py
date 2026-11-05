"""Je transforme les consensus UltraMedical exacts en exemples SFT anglais."""

import json
from pathlib import Path

from outils_sft import SYSTEME_EN, exemple


RACINE = Path(__file__).resolve().parents[1]
ANNOTATIONS = RACINE / "data/annotations/ultramedical_triage"
SORTIE = RACINE / "data/processed/sft_deux_phases"
PRIORITES = {
    "maximale": "highest urgency",
    "moderee": "moderate urgency",
    "differee": "delayed urgency",
}
ORIENTATIONS = {
    1: "Immediate life-saving care is required.",
    2: "Very rapid medical assessment is required because of the high risk.",
    3: "Medical assessment is needed and several resources will probably be required.",
    4: "Delayed assessment is appropriate; one resource category will probably be required.",
    5: "Non-urgent guidance is appropriate; no immediate additional resource is expected.",
}


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def preparer():
    """Construit un exemple par consensus exact et sa demande de traduction."""
    consensus = lire_jsonl(ANNOTATIONS / "consensus_retenus.jsonl")
    exemples = []
    demandes = []
    for vote in consensus:
        esi = vote["esi"]
        reponse = (
            f"ESI level: {esi}\n"
            f"Priority: {PRIORITES[vote['urgence']]}\n"
            f"Guidance: {ORIENTATIONS[esi]}\n"
            "This triage guidance does not replace assessment by a healthcare professional."
        )
        identifiant = vote["case_id"]
        exemples.append(exemple(
            identifiant,
            SYSTEME_EN,
            vote["texte_source"],
            reponse,
            langue="en",
            source="TsinghuaC3I/UltraMedical-Preference",
            source_split="train",
            type_tache="triage",
            licence="MIT",
            niveau_confiance="majorite_3_modeles_esi_exact",
            esi=esi,
            urgence=vote["urgence"],
            id_paire=identifiant,
        ))
        demandes.append({
            "id": identifiant,
            "texte_en": vote["texte_source"],
            "reponse_en": reponse,
        })

    if len(exemples) != len(consensus):
        raise ValueError("Un exemple SFT est attendu par consensus")
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "phase2_ultramedical_en.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in exemples)
    )
    (SORTIE / "traduction_ultra_entrees.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in demandes)
    )
    print(f"Exemples UltraMedical préparés : {len(exemples)}")
    return exemples


if __name__ == "__main__":
    preparer()
