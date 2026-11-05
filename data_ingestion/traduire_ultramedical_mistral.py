"""Je traduis les consensus UltraMedical de l'anglais vers le français par batch."""

import argparse
import json
import os
from pathlib import Path
import re
import urllib.request
import uuid

from annoter_pilote_mistral import MODELE, charger_cle


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/processed/sft_deux_phases"
SOURCE = BASE / "traduction_ultra_entrees.jsonl"
DOSSIER = BASE / "traduction_ultra_batch"
REQUETES = DOSSIER / "requetes.jsonl"
UPLOAD = DOSSIER / "upload.json"
JOB = DOSSIER / "job.json"
STATUT = DOSSIER / "statut.json"
BRUT = DOSSIER / "sortie_brute.jsonl"
SORTIE = BASE / "traduction_ultra_sorties.jsonl"
API = "https://api.mistral.ai/v1"
PRIX_ENTREE = 0.75 / 1_000_000
PRIX_SORTIE = 3.75 / 1_000_000
PLAFOND_USD = 0.80

CONSIGNE = """Translate the two English fields into natural clinical French.
Preserve every clinical fact, negation, number, unit, chronology and uncertainty.
Do not add, remove, interpret or correct medical information.
Preserve the exact ESI level. Translate the response headings and guidance.
Return one JSON object only:
{"id":"exact input id","texte_fr":"translation","reponse_fr":"translation"}
"""


def lire_jsonl(chemin):
    """Charge un fichier JSONL en ignorant les lignes vides."""
    return [json.loads(x) for x in chemin.read_text().splitlines() if x.strip()]


def ecrire_json(chemin, valeur):
    """Écrit un état JSON de façon atomique."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    temporaire.write_text(json.dumps(valeur, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporaire, chemin)


def preparer():
    """Construit les requêtes et bloque une estimation dépassant le budget."""
    entrees = lire_jsonl(SOURCE)
    if not entrees or len({x["id"] for x in entrees}) != len(entrees):
        raise ValueError("Entrées absentes ou identifiants dupliqués")
    DOSSIER.mkdir(parents=True, exist_ok=True)
    with REQUETES.open("w", encoding="utf-8") as fichier:
        for entree in entrees:
            body = {
                "messages": [
                    {"role": "system", "content": CONSIGNE},
                    {"role": "user", "content": json.dumps(entree, ensure_ascii=False)},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 900,
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
                "stream": False,
            }
            fichier.write(json.dumps({"custom_id": entree["id"], "body": body}, ensure_ascii=False) + "\n")
    octets = REQUETES.stat().st_size
    estimation = octets * 0.25 * PRIX_ENTREE + len(entrees) * 300 * PRIX_SORTIE
    manifeste = {
        "demandes": len(entrees),
        "estimation_usd": round(estimation, 4),
        "plafond_usd": PLAFOND_USD,
    }
    ecrire_json(DOSSIER / "manifeste.json", manifeste)
    if estimation > PLAFOND_USD:
        raise ValueError("Estimation supérieure au plafond")
    print(json.dumps(manifeste, ensure_ascii=False, indent=2))
    return entrees


def appel_json(methode, chemin, cle, donnees=None):
    """Appelle un endpoint JSON de l'API Mistral."""
    corps = None if donnees is None else json.dumps(donnees).encode()
    requete = urllib.request.Request(
        API + chemin,
        data=corps,
        method=methode,
        headers={"Authorization": "Bearer " + cle, **({"Content-Type": "application/json"} if corps else {})},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def televerser(cle):
    """Téléverse le JSONL des requêtes comme entrée du batch."""
    frontiere = "----ocr14-" + uuid.uuid4().hex
    corps = (
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"ultra-fr.jsonl\"\r\n"
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode() + REQUETES.read_bytes() + f"\r\n--{frontiere}--\r\n".encode()
    requete = urllib.request.Request(
        API + "/files",
        data=corps,
        method="POST",
        headers={"Authorization": "Bearer " + cle, "Content-Type": "multipart/form-data; boundary=" + frontiere},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def soumettre():
    """Prépare et soumet le batch une seule fois."""
    preparer()
    if JOB.exists():
        raise SystemExit("Job déjà enregistré : " + json.loads(JOB.read_text())["id"])
    cle = charger_cle()
    if not cle:
        raise SystemExit("Clé MISTRA_API_KEY ou MISTRAL_API_KEY absente")
    upload = televerser(cle)
    ecrire_json(UPLOAD, upload)
    job = appel_json("POST", "/batch/jobs", cle, {
        "input_files": [upload["id"]],
        "model": MODELE,
        "endpoint": "/v1/chat/completions",
        "timeout_hours": 24,
        "metadata": {"projet": "OCR_PROJET_14", "usage": "traduction_ultramedical_fr"},
    })
    ecrire_json(JOB, job)
    return job


def statut():
    """Récupère et mémorise l'état courant du batch."""
    job = json.loads(JOB.read_text())
    valeur = appel_json("GET", "/batch/jobs/" + job["id"], charger_cle())
    ecrire_json(STATUT, valeur)
    return valeur


def normaliser_reponse(valeur):
    """Convertit une réponse structurée en texte sans perdre ses champs."""
    if isinstance(valeur, str):
        return valeur.strip()
    if isinstance(valeur, dict):
        return "\n".join(
            f"{cle}: {contenu}" for cle, contenu in valeur.items()
            if contenu is not None and str(contenu).strip()
        )
    raise ValueError("Réponse traduite ni textuelle ni structurée")


def telecharger():
    """Télécharge puis valide toutes les traductions du batch terminé."""
    etat = statut()
    if etat.get("status") != "SUCCESS":
        raise SystemExit("Batch non terminé : " + str(etat.get("status")))
    requete = urllib.request.Request(
        API + f"/files/{etat['output_file']}/content",
        headers={"Authorization": "Bearer " + charger_cle()},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        BRUT.write_bytes(reponse.read())
    sources = {x["id"]: x for x in lire_jsonl(SOURCE)}
    niveaux_esi = {
        x["id"]: x["esi"] for x in lire_jsonl(BASE / "phase2_ultramedical_en.jsonl")
    }
    traductions = {}
    for ligne in lire_jsonl(BRUT):
        identifiant = ligne["custom_id"]
        contenu = ligne["response"]["body"]["choices"][0]["message"]["content"]
        traduction = json.loads(contenu)
        if traduction.get("id") != identifiant or set(traduction) != {"id", "texte_fr", "reponse_fr"}:
            raise ValueError("Traduction invalide : " + identifiant)
        traduction["texte_fr"] = traduction["texte_fr"].strip()
        traduction["reponse_fr"] = normaliser_reponse(traduction["reponse_fr"])
        if not traduction["texte_fr"] or not traduction["reponse_fr"]:
            raise ValueError("Traduction vide : " + identifiant)
        esi = niveaux_esi[identifiant]
        if not re.search(rf"\b(?:niveau[ _]+)?ESI\s*:\s*{esi}\b", traduction["reponse_fr"], re.I):
            raise ValueError("ESI absent : " + identifiant)
        traductions[identifiant] = traduction
    if set(traductions) != set(sources):
        raise ValueError(f"Traductions incomplètes : {len(traductions)}/{len(sources)}")
    SORTIE.write_text("".join(json.dumps(traductions[x], ensure_ascii=False) + "\n" for x in sources))
    print("Traductions validées :", len(traductions))


def afficher(valeur):
    """Affiche les compteurs utiles de l'état du batch."""
    champs = ("id", "status", "total_requests", "completed_requests", "succeeded_requests", "failed_requests")
    print(json.dumps({x: valeur.get(x) for x in champs}, indent=2))


def main():
    """Expose la préparation, la soumission, le suivi et le téléchargement."""
    parser = argparse.ArgumentParser(description=__doc__)
    groupe = parser.add_mutually_exclusive_group()
    groupe.add_argument("--submit", action="store_true")
    groupe.add_argument("--status", action="store_true")
    groupe.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.submit:
        afficher(soumettre())
    elif args.status:
        afficher(statut())
    elif args.download:
        telecharger()
    else:
        preparer()


if __name__ == "__main__":
    main()
