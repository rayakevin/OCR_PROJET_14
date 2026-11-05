"""Je prépare, lance et récupère le batch Mistral du corpus médical."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid

from annoter_pilote_mistral import (
    CONSIGNE_COMPACTE,
    MODELE,
    charger_cle,
    lire_jsonl,
    payload_compact,
    valider_vote_compact,
)


RACINE = Path(__file__).resolve().parents[1]
CORPUS = RACINE / "data/annotations/corpus/entrees/cas.jsonl"
PILOTE = RACINE / "data/annotations/arbitrage_mistral/pilote/reponses_compactes.jsonl"
DOSSIER = RACINE / "data/annotations/arbitrage_mistral/batch"
ENTREES = DOSSIER / "entrees.jsonl"
MANIFESTE = DOSSIER / "manifeste.json"
UPLOAD = DOSSIER / "upload.json"
JOB = DOSSIER / "job.json"
TENTATIVE = DOSSIER / "soumission_en_cours.json"
REJET = DOSSIER / "soumission_rejetee.json"
STATUT = DOSSIER / "statut.json"
SORTIE_BRUTE = DOSSIER / "sortie_brute.jsonl"
ERREURS_BRUTES = DOSSIER / "erreurs_brutes.jsonl"
SORTIE_COMPLETE = RACINE / "data/annotations/arbitrage_mistral/mistral.jsonl"
API = "https://api.mistral.ai/v1"
MAX_TOKENS = 180
PRIX_BATCH_ENTREE = 0.75 / 1_000_000
PRIX_BATCH_SORTIE = 3.75 / 1_000_000
DEPENSE_EXPERIMENTS_USD = 0.230889
PLAFOND_MISTRAL_USD = 4.80


def maintenant():
    """Retourne un horodatage UTC au format ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def ecrire_json_atomique(chemin, valeur):
    """Écrit un JSON via un fichier temporaire pour éviter un état partiel."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    temporaire.write_text(json.dumps(valeur, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporaire, chemin)


def sha256(chemin):
    """Calcule l’empreinte SHA-256 d’un fichier par blocs."""
    empreinte = hashlib.sha256()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            empreinte.update(bloc)
    return empreinte.hexdigest()


def votes_pilotes_valides(cas_par_id):
    """Récupère les 20 votes du pilote après validation de leur schéma."""
    votes = {}
    for ligne in lire_jsonl(PILOTE):
        case_id = ligne.get("case_id")
        if ligne.get("status") == "valide" and case_id in cas_par_id:
            votes[case_id] = valider_vote_compact(ligne["annotation"], cas_par_id[case_id])
    if len(votes) != 20:
        raise ValueError(f"20 votes pilotes attendus, {len(votes)} trouvés")
    return votes


def preparer():
    """Prépare les requêtes batch en réutilisant les votes pilotes déjà acquis."""
    cas = lire_jsonl(CORPUS)
    cas_par_id = {element["case_id"]: element for element in cas}
    if len(cas) != 2007 or len(cas_par_id) != len(cas):
        raise ValueError("Le corpus attendu doit contenir 2 007 identifiants uniques")
    pilotes = votes_pilotes_valides(cas_par_id)
    restants = [element for element in cas if element["case_id"] not in pilotes]
    DOSSIER.mkdir(parents=True, exist_ok=True)
    with ENTREES.open("w", encoding="utf-8") as fichier:
        for element in restants:
            corps = payload_compact(element)
            corps.pop("model", None)
            fichier.write(json.dumps({
                "custom_id": element["case_id"],
                "body": corps,
            }, ensure_ascii=False) + "\n")

    usages = [ligne["usage"] for ligne in lire_jsonl(PILOTE) if ligne.get("status") == "valide"]
    moyenne_entree = sum(u["prompt_tokens"] for u in usages) / len(usages)
    moyenne_sortie = sum(u["completion_tokens"] for u in usages) / len(usages)
    estimation = len(restants) * (
        moyenne_entree * PRIX_BATCH_ENTREE + moyenne_sortie * PRIX_BATCH_SORTIE
    )
    borne = len(restants) * (
        moyenne_entree * PRIX_BATCH_ENTREE + MAX_TOKENS * PRIX_BATCH_SORTIE
    )
    manifeste = {
        "cree_le": maintenant(),
        "modele": MODELE,
        "endpoint": "/v1/chat/completions",
        "raisonnement": "none",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_TOKENS,
        "nombre_corpus": len(cas),
        "nombre_reutilise_pilote": len(pilotes),
        "nombre_batch": len(restants),
        "case_ids": [element["case_id"] for element in restants],
        "sha256_entrees": sha256(ENTREES),
        "octets_entrees": ENTREES.stat().st_size,
        "moyenne_pilote_tokens_entree": moyenne_entree,
        "moyenne_pilote_tokens_sortie": moyenne_sortie,
        "cout_batch_estime_usd": round(estimation, 6),
        "cout_batch_borne_sortie_max_usd": round(borne, 6),
        "depense_mistral_avant_batch_usd": DEPENSE_EXPERIMENTS_USD,
        "cout_mistral_total_borne_usd": round(DEPENSE_EXPERIMENTS_USD + borne, 6),
        "sha256_consigne": hashlib.sha256(CONSIGNE_COMPACTE.encode()).hexdigest(),
    }
    ecrire_json_atomique(MANIFESTE, manifeste)
    return manifeste


def requete_json(methode, chemin, cle, donnees=None):
    """Appelle un endpoint JSON de l’API Mistral."""
    corps = None if donnees is None else json.dumps(donnees, ensure_ascii=False).encode()
    requete = urllib.request.Request(
        API + chemin,
        data=corps,
        headers={
            "Authorization": "Bearer " + cle,
            **({"Content-Type": "application/json"} if corps is not None else {}),
        },
        method=methode,
    )
    with urllib.request.urlopen(requete, timeout=180) as reponse:
        return json.load(reponse)


def televerser(cle):
    """Téléverse le fichier JSONL utilisé comme entrée du batch."""
    frontiere = "----codex-" + uuid.uuid4().hex
    contenu = ENTREES.read_bytes()
    morceaux = []
    for nom, valeur in (("purpose", "batch"),):
        morceaux.append(
            f"--{frontiere}\r\nContent-Disposition: form-data; name=\"{nom}\"\r\n\r\n{valeur}\r\n".encode()
        )
    morceaux.append(
        f"--{frontiere}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"entrees.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n".encode()
    )
    morceaux.extend((contenu, b"\r\n", f"--{frontiere}--\r\n".encode()))
    requete = urllib.request.Request(
        API + "/files",
        data=b"".join(morceaux),
        headers={
            "Authorization": "Bearer " + cle,
            "Content-Type": "multipart/form-data; boundary=" + frontiere,
        },
        method="POST",
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.load(reponse)


def soumettre():
    """Soumet le batch une seule fois et conserve les identifiants de reprise."""
    manifeste = preparer()
    if manifeste["cout_mistral_total_borne_usd"] > PLAFOND_MISTRAL_USD:
        raise SystemExit("Borne de coût supérieure au plafond local : aucun envoi")
    if JOB.exists():
        raise SystemExit("Un job est déjà enregistré : utiliser --status, sans nouvel envoi")
    if TENTATIVE.exists():
        raise SystemExit("Une tentative est déjà enregistrée : vérifier son issue avant tout nouvel envoi")
    cle = charger_cle()
    if not cle:
        raise SystemExit("Clé Mistral absente")

    if UPLOAD.exists():
        upload = json.loads(UPLOAD.read_text())
    else:
        upload = televerser(cle)
        ecrire_json_atomique(UPLOAD, upload)
    file_id = upload.get("id")
    if not file_id:
        raise ValueError("Identifiant du fichier téléversé absent")

    run_id = "ocr14-" + manifeste["sha256_entrees"][:16]
    ecrire_json_atomique(TENTATIVE, {
        "commencee_le": maintenant(), "run_id": run_id, "input_file_id": file_id
    })
    try:
        job = requete_json("POST", "/batch/jobs", cle, {
            "input_files": [file_id],
            "model": MODELE,
            "endpoint": "/v1/chat/completions",
            "timeout_hours": 24,
            "metadata": {"run_id": run_id, "projet": "OCR_PROJET_14"},
        })
    except urllib.error.HTTPError as erreur:
        detail = erreur.read().decode(errors="replace")[:2000]
        ecrire_json_atomique(REJET, {
            "date": maintenant(), "http_status": erreur.code, "detail": detail,
            "run_id": run_id, "input_file_id": file_id,
        })
        TENTATIVE.unlink(missing_ok=True)
        raise
    ecrire_json_atomique(JOB, job)
    TENTATIVE.unlink(missing_ok=True)
    return job


def obtenir_statut():
    """Récupère et mémorise l’état courant du batch."""
    if not JOB.exists():
        raise SystemExit("Aucun job enregistré")
    job_initial = json.loads(JOB.read_text())
    job_id = job_initial.get("id")
    statut = requete_json("GET", "/batch/jobs/" + job_id, charger_cle())
    ecrire_json_atomique(STATUT, statut)
    return statut


def telecharger_fichier(file_id, destination, cle):
    """Télécharge un fichier Mistral vers une destination locale."""
    requete = urllib.request.Request(
        API + f"/files/{file_id}/content",
        headers={"Authorization": "Bearer " + cle},
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        contenu = reponse.read()
    temporaire = destination.with_suffix(destination.suffix + ".tmp")
    temporaire.write_bytes(contenu)
    os.replace(temporaire, destination)


def extraire_annotation(ligne):
    """Extrait et désérialise l’annotation d’une ligne de résultat batch."""
    reponse = ligne.get("response") or {}
    corps = reponse.get("body") or {}
    contenu = corps["choices"][0]["message"]["content"]
    return json.loads(contenu)


def consolider_sortie():
    """Valide et réunit les votes pilotes et les résultats du batch."""
    cas = lire_jsonl(CORPUS)
    cas_par_id = {element["case_id"]: element for element in cas}
    annotations = votes_pilotes_valides(cas_par_id)
    erreurs = []
    for ligne in lire_jsonl(SORTIE_BRUTE):
        case_id = ligne.get("custom_id")
        try:
            annotations[case_id] = valider_vote_compact(extraire_annotation(ligne), cas_par_id[case_id])
        except Exception as erreur:  # les détails sont conservés pour reprise ciblée
            erreurs.append({"case_id": case_id, "erreur": str(erreur)})
    if erreurs or len(annotations) != len(cas):
        ecrire_json_atomique(DOSSIER / "erreurs_validation.json", erreurs)
        raise ValueError(f"Sortie incomplète : {len(annotations)}/{len(cas)} valides, {len(erreurs)} erreurs")
    with SORTIE_COMPLETE.open("w", encoding="utf-8") as fichier:
        for element in cas:
            fichier.write(json.dumps(annotations[element["case_id"]], ensure_ascii=False) + "\n")
    return len(annotations)


def telecharger():
    """Télécharge les sorties d’un batch terminé puis lance leur consolidation."""
    statut = obtenir_statut()
    if statut.get("status") != "SUCCESS":
        raise SystemExit("Le job n'est pas terminé avec succès : " + str(statut.get("status")))
    cle = charger_cle()
    if statut.get("output_file"):
        telecharger_fichier(statut["output_file"], SORTIE_BRUTE, cle)
    if statut.get("error_file"):
        telecharger_fichier(statut["error_file"], ERREURS_BRUTES, cle)
    return consolider_sortie()


def afficher(statut):
    """Affiche uniquement les indicateurs utiles du batch."""
    champs = ("id", "status", "total_requests", "completed_requests", "succeeded_requests", "failed_requests")
    print(json.dumps({champ: statut.get(champ) for champ in champs}, ensure_ascii=False, indent=2))


def main():
    """Expose les étapes préparer, soumettre, suivre et télécharger en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    groupe = parser.add_mutually_exclusive_group()
    groupe.add_argument("--submit", action="store_true")
    groupe.add_argument("--status", action="store_true")
    groupe.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        if args.submit:
            afficher(soumettre())
        elif args.status:
            afficher(obtenir_statut())
        elif args.download:
            print(f"Annotations consolidées : {telecharger()}")
        else:
            print(json.dumps(preparer(), ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as erreur:
        detail = erreur.read().decode(errors="replace")[:2000]
        print(f"HTTP {erreur.code}: {detail}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
