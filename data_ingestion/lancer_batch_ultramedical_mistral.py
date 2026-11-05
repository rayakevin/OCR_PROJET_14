"""Je prépare, soumets et récupère le troisième vote Mistral pour UltraMedical."""

import argparse
import json
import os
from pathlib import Path
import urllib.request
import uuid

from annoter_pilote_mistral import (
    MODELE,
    charger_cle,
    payload_compact,
    valider_vote_compact,
)


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/annotations/ultramedical_triage"
SOURCE = BASE / "entrees/cas.jsonl"
DOSSIER = BASE / "mistral_batch"
REQUETES = DOSSIER / "requetes.jsonl"
UPLOAD = DOSSIER / "upload.json"
JOB = DOSSIER / "job.json"
STATUT = DOSSIER / "statut.json"
BRUT = DOSSIER / "sortie_brute.jsonl"
SORTIE = BASE / "mistral"
API = "https://api.mistral.ai/v1"

# Tarifs batch utilisés pour la borne avant soumission.
PRIX_ENTREE = 0.75 / 1_000_000
PRIX_SORTIE = 3.75 / 1_000_000
MAX_TOKENS = 180
PLAFOND_BATCH_USD = 1.75


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(x) for x in chemin.read_text().splitlines() if x.strip()]


def ecrire_json(chemin, valeur):
    """Écrit un état JSON de façon atomique."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    temporaire.write_text(json.dumps(valeur, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporaire, chemin)


def preparer():
    """Prépare seulement les cas où le troisième vote peut changer le consensus."""
    cas = lire_jsonl(SOURCE)
    if len(cas) != 2000 or len({x["case_id"] for x in cas}) != 2000:
        raise ValueError("2 000 cas UltraMedical uniques attendus")
    def annotations(modele):
        lignes = []
        for chemin in sorted((BASE / modele).glob("lot-*.jsonl")):
            lignes.extend(lire_jsonl(chemin))
        if len(lignes) != len(cas):
            raise ValueError(f"2 000 annotations {modele} attendues")
        return {x["case_id"]: x for x in lignes}

    codex, claude = annotations("codex"), annotations("claude")
    demandes = []
    for element in cas:
        gauche, droite = codex[element["case_id"]], claude[element["case_id"]]
        meme_decision = gauche["decision_entree"] == droite["decision_entree"]
        non_candidat_fixe = meme_decision and gauche["decision_entree"] != "candidat_initial"
        classement_fixe = (
            meme_decision and gauche["decision_entree"] == "candidat_initial"
            and gauche["statut_classement"] == droite["statut_classement"] == "classe"
            and gauche["urgence"] == droite["urgence"] and gauche["esi"] == droite["esi"]
        )
        if not (non_candidat_fixe or classement_fixe):
            demandes.append(element)

    DOSSIER.mkdir(parents=True, exist_ok=True)
    octets = 0
    with REQUETES.open("w", encoding="utf-8") as fichier:
        for element in demandes:
            corps = payload_compact(element)
            corps.pop("model", None)
            corps["max_tokens"] = MAX_TOKENS
            ligne = json.dumps({"custom_id": element["case_id"], "body": corps}, ensure_ascii=False)
            fichier.write(ligne + "\n")
            octets += len(ligne.encode())
    # Ratio prudent, calibré sur le batch MediQAl déjà terminé : 0,243 jeton/octet.
    estimation = octets * 0.25 * PRIX_ENTREE + len(demandes) * 130 * PRIX_SORTIE
    manifeste = {
        "cas_total": len(cas), "majorites_deja_fixees": len(cas) - len(demandes),
        "votes_mistral_demandes": len(demandes), "estimation_usd": round(estimation, 4),
        "plafond_usd": PLAFOND_BATCH_USD,
    }
    ecrire_json(DOSSIER / "manifeste.json", manifeste)
    print(json.dumps(manifeste, ensure_ascii=False, indent=2))
    return demandes, estimation


def televerser(cle):
    """Téléverse le JSONL comme fichier d'entrée batch."""
    frontiere = "----ocr14-" + uuid.uuid4().hex
    contenu = REQUETES.read_bytes()
    corps = (
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"ultramedical.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n"
    ).encode() + contenu + f"\r\n--{frontiere}--\r\n".encode()
    requete = urllib.request.Request(
        API + "/files", data=corps, method="POST",
        headers={"Authorization": "Bearer " + cle,
                 "Content-Type": "multipart/form-data; boundary=" + frontiere},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def appel_json(methode, chemin, cle, donnees=None):
    """Appelle un endpoint JSON de l'API Mistral."""
    corps = None if donnees is None else json.dumps(donnees).encode()
    requete = urllib.request.Request(
        API + chemin, data=corps, method=methode,
        headers={"Authorization": "Bearer " + cle,
                 **({"Content-Type": "application/json"} if corps else {})},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def soumettre():
    """Prépare puis soumet le batch une seule fois."""
    _, borne = preparer()
    if borne > PLAFOND_BATCH_USD:
        raise SystemExit(f"Estimation supérieure à {PLAFOND_BATCH_USD:.2f} USD : soumission bloquée")
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
        "metadata": {"projet": "OCR_PROJET_14", "usage": "vote_ultramedical"},
    })
    ecrire_json(JOB, job)
    return job


def statut():
    """Récupère et mémorise l'état courant du batch."""
    job = json.loads(JOB.read_text())
    valeur = appel_json("GET", "/batch/jobs/" + job["id"], charger_cle())
    ecrire_json(STATUT, valeur)
    return valeur


def telecharger():
    """Télécharge et valide les 2 000 votes d'un batch terminé."""
    etat = statut()
    if etat.get("status") != "SUCCESS":
        raise SystemExit("Batch non terminé : " + str(etat.get("status")))
    requete = urllib.request.Request(
        API + f"/files/{etat['output_file']}/content",
        headers={"Authorization": "Bearer " + charger_cle()},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        BRUT.write_bytes(reponse.read())
    cas = {x["case_id"]: x for x in lire_jsonl(SOURCE)}
    annotations = {}
    for ligne in lire_jsonl(BRUT):
        identifiant = ligne["custom_id"]
        contenu = ligne["response"]["body"]["choices"][0]["message"]["content"]
        annotations[identifiant] = valider_vote_compact(json.loads(contenu), cas[identifiant])
    demandes = {x["custom_id"] for x in lire_jsonl(REQUETES)}
    if set(annotations) != demandes:
        raise ValueError(f"Votes incomplets : {len(annotations)}/{len(demandes)}")
    SORTIE.mkdir(parents=True, exist_ok=True)
    ordre = [x for x in cas if x in annotations]
    (SORTIE / "votes.jsonl").write_text(
        "".join(json.dumps(annotations[x], ensure_ascii=False) + "\n" for x in ordre)
    )
    print("Votes Mistral validés :", len(annotations))


def afficher(valeur):
    """Affiche seulement les compteurs utiles du batch."""
    champs = ("id", "status", "total_requests", "completed_requests",
              "succeeded_requests", "failed_requests")
    print(json.dumps({x: valeur.get(x) for x in champs}, indent=2))


def main():
    """Expose la préparation, la soumission, le suivi et le téléchargement."""
    parseur = argparse.ArgumentParser(description=__doc__)
    groupe = parseur.add_mutually_exclusive_group()
    groupe.add_argument("--submit", action="store_true")
    groupe.add_argument("--status", action="store_true")
    groupe.add_argument("--download", action="store_true")
    arguments = parseur.parse_args()
    if arguments.submit:
        afficher(soumettre())
    elif arguments.status:
        afficher(statut())
    elif arguments.download:
        telecharger()
    else:
        preparer()


if __name__ == "__main__":
    main()
