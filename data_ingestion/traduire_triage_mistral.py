"""Je traduis en anglais les 700 paires de triage avec Mistral Batch."""

import argparse
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import uuid

from annoter_pilote_mistral import MODELE, charger_cle
from outils_sft import SYSTEME_EN, ecrire_jsonl


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/processed/sft_deux_phases"
ETAT = BASE / "traduction_batch"
SOURCE = BASE / "traduction_triage_entrees.jsonl"
REQUETES = ETAT / "requetes.jsonl"
UPLOAD = ETAT / "upload.json"
JOB = ETAT / "job.json"
STATUT = ETAT / "statut.json"
BRUT = ETAT / "sortie_brute.jsonl"
API = "https://api.mistral.ai/v1"

CONSIGNE = """Translate the two French fields into natural clinical English.
Preserve every clinical fact, negation, number, unit, chronology and uncertainty.
Do not add, remove, interpret or correct medical information.
Preserve the exact ESI level. Translate the response headings and guidance.
Return one JSON object only:
{"id":"exact input id","texte_en":"translation","reponse_en":"translation"}
"""

# Trois sorties ont interprété ou corrigé la source malgré la consigne. Ces
# remplacements documentent la correction sans modifier la sortie brute.
CORRECTIONS_TEXTE = {
    "mediqal-6d6f5aefc2c0f590547fb09d873ad2052e06d820103cf2d6b1abf08098603a9f": (
        "a 30-pack-year smoking history (1/2 pack per day for 30 years)",
        "a smoking history of 1/2 pack per day for 30 years",
    ),
    "mediqal-a9247d7d6ee7fa965cad986f6a131d96bcf2c403f5779a24e7572e47a4e0a812": (
        "PEF = 200 L/min",
        "PEF = 2001/min",
    ),
    "mediqal-9089ce60e0ac220385d88af1638f8882aabf6762915ffba6e74d10c7bbaf0389": (
        "37.6°C",
        "37.F°C",
    ),
}


def lire_jsonl(chemin):
    """Charge un fichier JSONL en ignorant les lignes vides."""
    return [json.loads(x) for x in chemin.read_text().splitlines() if x.strip()]


def ecrire_json(chemin, valeur):
    """Écrit un état JSON de manière atomique."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    temporaire.write_text(json.dumps(valeur, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporaire, chemin)


def preparer():
    """Transforme les 700 paires françaises en requêtes de traduction batch."""
    entrees = lire_jsonl(SOURCE)
    if len(entrees) != 700 or len({x["id"] for x in entrees}) != 700:
        raise ValueError("700 entrées uniques attendues")
    ETAT.mkdir(parents=True, exist_ok=True)
    with REQUETES.open("w", encoding="utf-8") as fichier:
        for entree in entrees:
            body = {
                "messages": [
                    {"role": "system", "content": CONSIGNE},
                    {"role": "user", "content": json.dumps({
                        "id": entree["id"],
                        "texte_fr": entree["texte_fr"],
                        "reponse_fr": entree["reponse_fr"],
                    }, ensure_ascii=False)},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 900,
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
                "stream": False,
            }
            fichier.write(json.dumps({"custom_id": entree["id"], "body": body}, ensure_ascii=False) + "\n")
    return entrees


def appel_json(methode, chemin, cle, donnees=None):
    """Appelle un endpoint JSON de l’API Mistral."""
    corps = None if donnees is None else json.dumps(donnees).encode()
    requete = urllib.request.Request(
        API + chemin, data=corps, method=methode,
        headers={"Authorization": "Bearer " + cle, **({"Content-Type": "application/json"} if corps else {})},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def televerser(cle):
    """Téléverse le fichier de requêtes de traduction."""
    frontiere = "----ocr14-" + uuid.uuid4().hex
    contenu = REQUETES.read_bytes()
    corps = (
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"traductions.jsonl\"\r\n"
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode() + contenu + f"\r\n--{frontiere}--\r\n".encode()
    requete = urllib.request.Request(
        API + "/files", data=corps, method="POST",
        headers={"Authorization": "Bearer " + cle, "Content-Type": "multipart/form-data; boundary=" + frontiere},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def soumettre():
    """Crée le batch de traduction et conserve son identifiant."""
    preparer()
    if JOB.exists():
        raise SystemExit("Job déjà enregistré")
    cle = charger_cle()
    upload = json.loads(UPLOAD.read_text()) if UPLOAD.exists() else televerser(cle)
    if not UPLOAD.exists():
        ecrire_json(UPLOAD, upload)
    job = appel_json("POST", "/batch/jobs", cle, {
        "input_files": [upload["id"]],
        "model": MODELE,
        "endpoint": "/v1/chat/completions",
        "timeout_hours": 24,
        "metadata": {"projet": "OCR_PROJET_14", "usage": "traduction_triage"},
    })
    ecrire_json(JOB, job)
    return job


def statut():
    """Récupère et mémorise l’état actuel du batch."""
    job = json.loads(JOB.read_text())
    resultat = appel_json("GET", "/batch/jobs/" + job["id"], charger_cle())
    ecrire_json(STATUT, resultat)
    return resultat


def telecharger_fichier(file_id):
    """Télécharge le fichier de sortie du batch terminé."""
    requete = urllib.request.Request(
        API + f"/files/{file_id}/content", headers={"Authorization": "Bearer " + charger_cle()}
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        BRUT.write_bytes(reponse.read())


def contenu(ligne):
    """Extrait l’objet JSON produit pour une ligne du batch."""
    return json.loads(ligne["response"]["body"]["choices"][0]["message"]["content"])


def normaliser_reponse(valeur):
    """Convertit une réponse structurée en texte sans supprimer ses champs."""
    if isinstance(valeur, str):
        return valeur.strip()
    if isinstance(valeur, dict):
        return "\n".join(
            f"{cle}: {contenu}" for cle, contenu in valeur.items()
            if contenu is not None and str(contenu).strip()
        )
    raise ValueError("Réponse traduite ni textuelle ni structurée")


def corriger_texte(identifiant, texte):
    """Applique une correction revue manuellement et vérifie le texte attendu."""
    if identifiant not in CORRECTIONS_TEXTE:
        return texte
    ancien, nouveau = CORRECTIONS_TEXTE[identifiant]
    if ancien not in texte:
        raise ValueError("Passage à corriger absent : " + identifiant)
    return texte.replace(ancien, nouveau, 1)


def finaliser(telecharger=True):
    """Valide les traductions et construit les paires bilingues par split."""
    if telecharger:
        etat = statut()
        if etat.get("status") != "SUCCESS":
            raise SystemExit("Job non terminé : " + str(etat.get("status")))
        telecharger_fichier(etat["output_file"])
    elif not BRUT.exists():
        raise FileNotFoundError("Sortie batch locale absente")
    sources = {x["id"]: x for x in lire_jsonl(SOURCE)}
    traductions = {}
    for ligne in lire_jsonl(BRUT):
        traduction = contenu(ligne)
        identifiant = ligne["custom_id"]
        if traduction.get("id") != identifiant or set(traduction) != {"id", "texte_en", "reponse_en"}:
            raise ValueError("Traduction invalide : " + identifiant)
        traduction["texte_en"] = corriger_texte(identifiant, traduction["texte_en"].strip())
        traduction["reponse_en"] = normaliser_reponse(traduction["reponse_en"])
        if not traduction["texte_en"] or not traduction["reponse_en"]:
            raise ValueError("Traduction vide : " + identifiant)
        esi = sources[identifiant]["esi"]
        if not re.search(rf"\bESI(?: level)?\s*:\s*{esi}\b", traduction["reponse_en"], re.I):
            raise ValueError("ESI absent de la traduction : " + identifiant)
        traductions[identifiant] = traduction
    if len(traductions) != 700:
        raise ValueError(f"700 traductions attendues, {len(traductions)} obtenues")

    for split in ("train", "validation"):
        fr = lire_jsonl(BASE / f"phase2_triage_fr_{split}.jsonl")
        en = []
        for ligne in fr:
            traduction = traductions[ligne["id"]]
            copie = ligne | {
                "id": ligne["id"] + "-en",
                "langue": "en",
                "niveau_confiance": ligne["niveau_confiance"] + "+traduction_mistral",
                "id_paire": ligne["id"],
                "messages": [
                    {"role": "system", "content": SYSTEME_EN},
                    {"role": "user", "content": traduction["texte_en"]},
                    {"role": "assistant", "content": traduction["reponse_en"]},
                ],
            }
            ligne["id_paire"] = ligne["id"]
            en.append(copie)
        ecrire_jsonl(BASE / f"phase2_mediqal_{split}.jsonl", fr + en)
    return 700


def afficher(valeur):
    """Affiche les compteurs utiles de l’état du batch."""
    champs = ("id", "status", "total_requests", "completed_requests", "succeeded_requests", "failed_requests")
    print(json.dumps({x: valeur.get(x) for x in champs}, indent=2))


def main():
    """Expose la préparation, la soumission, le suivi et la finalisation."""
    parser = argparse.ArgumentParser(description=__doc__)
    groupe = parser.add_mutually_exclusive_group()
    groupe.add_argument("--submit", action="store_true")
    groupe.add_argument("--status", action="store_true")
    groupe.add_argument("--download", action="store_true")
    groupe.add_argument("--finalize-local", action="store_true")
    args = parser.parse_args()
    if args.submit:
        afficher(soumettre())
    elif args.status:
        afficher(statut())
    elif args.download:
        print("Traductions validées :", finaliser())
    elif args.finalize_local:
        print("Traductions locales validées :", finaliser(telecharger=False))
    else:
        print("Requêtes préparées :", len(preparer()))


if __name__ == "__main__":
    main()
