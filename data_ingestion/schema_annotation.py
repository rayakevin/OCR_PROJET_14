"""Schéma commun et contrôles des annotations de triage."""

import hashlib
import json
from pathlib import Path


RACINE = Path(__file__).resolve().parents[1]
VERSION_CONSIGNE = "qualification_triage_v2_budget"


def fingerprint(valeur):
    """Calcule une empreinte stable d’un objet JSON sérialisable."""
    return hashlib.sha256(
        json.dumps(valeur, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def lire_consigne():
    """Retourne la version et le texte de la consigne d’annotation."""
    chemin = Path(__file__).with_name("consigne_annotation.txt")
    return VERSION_CONSIGNE, chemin.read_text(encoding="utf-8").strip()


def valider(data, case):
    """Vérifie le schéma, les citations et les règles métier d’une annotation."""
    enums = {
        "presentation_patient": {"oui", "non", "a_revoir"},
        "temporalite": {"initial", "ulterieur", "mixte", "indetermine"},
        "decision_entree": {"candidat_initial", "a_adapter", "a_ecarter", "a_revoir"},
        "statut_classement": {"classe", "insuffisant", "non_classe"},
    }
    if not isinstance(data, dict) or data.get("case_id") != case["case_id"]:
        raise ValueError("Objet JSON ou identifiant incorrect")
    for champ, choix in enums.items():
        if not isinstance(data.get(champ), str) or data[champ] not in choix:
            raise ValueError("Valeur incorrecte : " + champ)
    for champ in ("justification_qualification", "justification_classement"):
        if not isinstance(data.get(champ), str) or not 1 <= len(data[champ].split()) <= 40:
            raise ValueError("Justification vide ou trop longue : " + champ)
    for champ in ("extraits_qualification", "informations_a_preciser"):
        if not isinstance(data.get(champ), list) or any(
            not isinstance(x, str) or not x.strip() for x in data[champ]
        ):
            raise ValueError("Liste de textes attendue : " + champ)
    if not data["extraits_qualification"]:
        raise ValueError("Qualification sans extrait")
    extraits = list(data["extraits_qualification"])
    if not isinstance(data.get("signes_alerte"), list):
        raise ValueError("Liste de signes attendue")
    for signe in data["signes_alerte"]:
        if (
            not isinstance(signe, dict)
            or not isinstance(signe.get("signe"), str)
            or not signe["signe"].strip()
            or not isinstance(signe.get("extrait"), str)
        ):
            raise ValueError("Signe d'alerte incorrect")
        extraits.append(signe["extrait"])
    if any(not extrait.strip() or extrait not in case["texte_source"] for extrait in extraits):
        raise ValueError("Extrait absent du texte source")

    candidat = data["decision_entree"] == "candidat_initial"
    if candidat and data["presentation_patient"] != "oui":
        raise ValueError("Candidat sans présentation de patient")
    if not candidat and data["statut_classement"] != "non_classe":
        raise ValueError("Classement interdit pour cette entrée")
    if candidat and data["statut_classement"] == "non_classe":
        raise ValueError("Candidat : classement ou insuffisance attendus")
    if data["statut_classement"] == "classe":
        correspondance = {
            1: "maximale",
            2: "maximale",
            3: "moderee",
            4: "differee",
            5: "differee",
        }
        if (
            type(data.get("esi")) is not int
            or data["esi"] not in correspondance
            or data.get("urgence") != correspondance[data["esi"]]
        ):
            raise ValueError("Incohérence ESI/urgence")
    elif (
        "esi" not in data
        or "urgence" not in data
        or data["esi"] is not None
        or data["urgence"] is not None
    ):
        raise ValueError("Abstention : esi et urgence doivent être null")
    return data
