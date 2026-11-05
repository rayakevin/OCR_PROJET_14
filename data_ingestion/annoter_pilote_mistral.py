"""Je lance un pilote Mistral de 20 arbitrages avec suivi du coût et reprise sûre."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from schema_annotation import fingerprint, valider


RACINE = Path(__file__).resolve().parents[1]
CORPUS = RACINE / "data/annotations/corpus"
ARBITRAGE = RACINE / "data/annotations/arbitrage_mistral"
PILOTE = ARBITRAGE / "pilote"
MODELE = "mistral-medium-3-5"
ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MAX_TOKENS = 350
BUDGET_USD = 0.35
PRIX_ENTREE = 1.5 / 1_000_000
PRIX_SORTIE = 7.5 / 1_000_000
REPARTITION = {
    "qualification_en_desaccord": 7,
    "urgence_en_desaccord": 7,
    "abstention_de_classement": 6,
}

CONSIGNE_COMPACTE = """Tu es un annotateur indépendant de cas médicaux pédagogiques.
Ta tâche est de qualifier la présentation au premier accueil puis, seulement si elle est exploitable, d'attribuer un niveau ESI et une urgence.

QUALIFICATION
- Utilise seulement les informations explicitement disponibles au premier accueil.
- Un antécédent ou traitement déjà connu peut être utilisé.
- Un examen, résultat, diagnostic ou événement explicitement ultérieur ne doit pas être utilisé pour classer l'état initial.
- Si le récit mélange plusieurs temps, choisis a_adapter.
- Si la disponibilité temporelle d'une information déterminante est incertaine, choisis a_revoir.
- Si le texte n'est pas une présentation de patient, choisis a_ecarter.
- Une consultation chronique ou programmée peut rester candidat_initial.

CLASSEMENT
- Classe uniquement si decision_entree=candidat_initial.
- ESI 1 : intervention vitale immédiate.
- ESI 2 : haut risque, confusion aiguë, douleur ou détresse sévère.
- ESI 3 : plusieurs catégories de ressources attendues.
- ESI 4 : une catégorie de ressource attendue.
- ESI 5 : aucune ressource attendue.
- Urgence maximale = ESI 1 ou 2 ; moderee = ESI 3 ; differee = ESI 4 ou 5.
- Une constante anormale impose une réévaluation, pas automatiquement ESI 2.
- Une information absente n'est pas normale.
- Si le cas est candidat mais insuffisant pour décider, statut_classement=insuffisant et esi/urgence=null.
- Si le cas n'est pas candidat_initial, statut_classement=non_classe et esi/urgence=null.

Réponds par un unique objet JSON, sans commentaire :
{
  "case_id": "identifiant exact",
  "presentation_patient": "oui|non|a_revoir",
  "temporalite": "initial|ulterieur|mixte|indetermine",
  "decision_entree": "candidat_initial|a_adapter|a_ecarter|a_revoir",
  "statut_classement": "classe|insuffisant|non_classe",
  "esi": "entier 1 à 5 ou null",
  "urgence": "maximale|moderee|differee ou null"
}
"""


def lire_jsonl(chemin):
    """Charge un fichier JSONL en ignorant les lignes vides."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def charger_cle():
    """Lit la clé Mistral depuis l’environnement ou le fichier .env local."""
    for nom in ("MISTRA_API_KEY", "MISTRAL_API_KEY"):
        if os.environ.get(nom):
            return os.environ[nom]
    chemin = RACINE / ".env"
    if chemin.exists():
        for ligne in chemin.read_text().splitlines():
            nom, separateur, valeur = ligne.strip().removeprefix("export ").partition("=")
            if separateur and nom.strip() in {"MISTRA_API_KEY", "MISTRAL_API_KEY"}:
                valeur = valeur.strip()
                if len(valeur) >= 2 and valeur[0] == valeur[-1] and valeur[0] in "\"'":
                    valeur = valeur[1:-1]
                return valeur
    return ""


def selectionner_cas():
    """Construit un échantillon pilote stratifié parmi les désaccords du consensus."""
    sources = {cas["case_id"]: cas for cas in lire_jsonl(CORPUS / "entrees/cas.jsonl")}
    resultats = lire_jsonl(CORPUS / "consensus.jsonl")
    rng = random.Random(42)
    selection = []
    for motif, nombre in REPARTITION.items():
        candidats = [resultat for resultat in resultats if resultat["motif"] == motif]
        rng.shuffle(candidats)
        selection.extend((sources[resultat["case_id"]] | {"motif_selection": motif}) for resultat in candidats[:nombre])
    rng.shuffle(selection)
    if len(selection) != 20 or len({cas["case_id"] for cas in selection}) != 20:
        raise ValueError("Le pilote doit contenir 20 cas litigieux uniques")
    return selection


def preparer_entrees(cas, consigne):
    """Enregistre les cas du pilote, leur consigne et un manifeste de traçabilité."""
    entrees = PILOTE / "entrees"
    entrees.mkdir(parents=True, exist_ok=True)
    (entrees / "cas.jsonl").write_text(
        "".join(json.dumps(element, ensure_ascii=False) + "\n" for element in cas)
    )
    (entrees / "consigne_commune.txt").write_text(consigne)
    manifeste = {
        "modele": MODELE,
        "raisonnement": "none",
        "max_tokens": MAX_TOKENS,
        "nombre_cas": len(cas),
        "repartition": dict(Counter(element["motif_selection"] for element in cas)),
        "case_ids": [element["case_id"] for element in cas],
        "sha256_consigne": fingerprint(consigne),
    }
    (entrees / "manifeste.json").write_text(json.dumps(manifeste, ensure_ascii=False, indent=2) + "\n")


def payload(consigne, cas):
    """Construit une requête Mistral avec le schéma d’annotation complet."""
    return {
        "model": MODELE,
        "messages": [
            {"role": "system", "content": consigne},
            {"role": "user", "content": json.dumps({
                "case_id": cas["case_id"],
                "texte_source": cas["texte_source"],
            }, ensure_ascii=False)},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def payload_compact(cas):
    """Construit une requête limitée aux champs nécessaires au troisième vote."""
    return {
        "model": MODELE,
        "messages": [
            {"role": "system", "content": CONSIGNE_COMPACTE},
            {"role": "user", "content": json.dumps({
                "case_id": cas["case_id"],
                "texte_source": cas["texte_source"],
            }, ensure_ascii=False)},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 180,
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def valider_vote_compact(data, cas):
    """Contrôle les valeurs, les nullités et la cohérence ESI/urgence d’un vote."""
    enums = {
        "presentation_patient": {"oui", "non", "a_revoir"},
        "temporalite": {"initial", "ulterieur", "mixte", "indetermine"},
        "decision_entree": {"candidat_initial", "a_adapter", "a_ecarter", "a_revoir"},
        "statut_classement": {"classe", "insuffisant", "non_classe"},
    }
    if not isinstance(data, dict) or data.get("case_id") != cas["case_id"]:
        raise ValueError("Objet JSON ou identifiant incorrect")
    if set(data) != {"case_id", *enums, "esi", "urgence"}:
        raise ValueError("Champs compacts incorrects")
    for champ, choix in enums.items():
        if data.get(champ) not in choix:
            raise ValueError("Valeur incorrecte : " + champ)
    candidat = data["decision_entree"] == "candidat_initial"
    if candidat and data["presentation_patient"] != "oui":
        raise ValueError("Candidat sans présentation de patient")
    if not candidat and data["statut_classement"] != "non_classe":
        raise ValueError("Classement interdit pour cette entrée")
    if candidat and data["statut_classement"] == "non_classe":
        raise ValueError("Candidat : classement ou insuffisance attendu")
    if data["statut_classement"] == "classe":
        correspondance = {1: "maximale", 2: "maximale", 3: "moderee", 4: "differee", 5: "differee"}
        if type(data.get("esi")) is not int or data["esi"] not in correspondance:
            raise ValueError("ESI incorrect")
        if data.get("urgence") != correspondance[data["esi"]]:
            raise ValueError("Incohérence ESI/urgence")
    elif data.get("esi") is not None or data.get("urgence") is not None:
        raise ValueError("Abstention : esi et urgence doivent être null")
    return data


def reserve(payload_requete):
    """Calcule une borne haute du coût d’une requête avant son envoi."""
    octets = len(json.dumps(payload_requete, ensure_ascii=False).encode())
    return octets * PRIX_ENTREE + MAX_TOKENS * PRIX_SORTIE


def extraire_contenu(reponse):
    """Extrait le texte seulement si la génération Mistral est terminée normalement."""
    contenu = reponse["choices"][0]["message"]["content"]
    if isinstance(contenu, str):
        return contenu
    if isinstance(contenu, list):
        textes = []
        for bloc in contenu:
            if isinstance(bloc, dict) and bloc.get("type") == "text":
                textes.append(bloc.get("text", ""))
        return "".join(textes)
    raise ValueError("Contenu Mistral inattendu")


def cout_usage(reponse):
    """Calcule le coût réel à partir des jetons déclarés par l’API."""
    usage = reponse.get("usage") or {}
    entree = usage.get("prompt_tokens")
    sortie = usage.get("completion_tokens")
    if type(entree) is not int or type(sortie) is not int:
        return None
    return entree * PRIX_ENTREE + sortie * PRIX_SORTIE


def appeler(payload_requete, cle):
    """Envoie une requête synchrone à l’API Mistral."""
    requete = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload_requete, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + cle, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(requete, timeout=180) as reponse:
        return json.load(reponse)


def main():
    """Prépare ou exécute le pilote, avec reprise et plafond de dépense."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Effectuer les appels Mistral payants")
    parser.add_argument("--compact", action="store_true", help="Tester le vote compact sans citations")
    args = parser.parse_args()

    consigne = (CORPUS / "entrees/consigne_commune.txt").read_text()
    cas = selectionner_cas()
    preparer_entrees(cas, consigne)
    requetes = [(element, payload_compact(element) if args.compact else payload(consigne, element)) for element in cas]
    reserve_totale = sum(reserve(requete) for _, requete in requetes)
    print(f"Pilote : {len(cas)} cas ; réserve maximale : {reserve_totale:.4f} USD ; plafond : {BUDGET_USD:.2f} USD")
    print("Répartition :", dict(Counter(element["motif_selection"] for element in cas)))
    if reserve_totale > BUDGET_USD:
        raise SystemExit("Réserve supérieure au plafond : aucun appel envoyé")
    if not args.run:
        print("Prévisualisation terminée : aucun appel réseau")
        return

    cle = charger_cle()
    if not cle:
        raise SystemExit("Clé absente : MISTRA_API_KEY ou MISTRAL_API_KEY")
    PILOTE.mkdir(parents=True, exist_ok=True)
    journal = PILOTE / ("reponses_compactes.jsonl" if args.compact else "reponses_none.jsonl")
    existantes = {}
    if journal.exists():
        for ligne in journal.read_text().splitlines():
            if ligne.strip():
                entree = json.loads(ligne)
                existantes[entree["case_id"]] = entree

    depense = sum(entree.get("cout_usd") or entree.get("budget_impute_usd", 0) for entree in existantes.values())
    for element, payload_requete in requetes:
        case_id = element["case_id"]
        if case_id in existantes:
            continue
        reservation = reserve(payload_requete)
        if depense + reservation > BUDGET_USD:
            print("ARRÊT BUDGET avant", case_id)
            break
        ligne = {
            "case_id": case_id,
            "provider": "mistral",
            "model_demande": MODELE,
            "raisonnement": "none",
            "max_tokens": MAX_TOKENS,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "status": "erreur_api",
            "budget_impute_usd": reservation,
        }
        debut = time.monotonic()
        try:
            reponse = appeler(payload_requete, cle)
            ligne["usage"] = reponse.get("usage")
            ligne["model_retourne"] = reponse.get("model")
            ligne["finish_reason"] = reponse["choices"][0].get("finish_reason")
            ligne["reponse_brute"] = reponse
            cout = cout_usage(reponse)
            if cout is not None:
                ligne["cout_usd"] = cout
                ligne["budget_impute_usd"] = cout
            annotation = json.loads(extraire_contenu(reponse))
            ligne["annotation"] = valider_vote_compact(annotation, element) if args.compact else valider(annotation, element)
            ligne["status"] = "valide"
        except urllib.error.HTTPError as erreur:
            ligne["erreur"] = f"HTTP {erreur.code}"
        except (urllib.error.URLError, TimeoutError, OSError):
            ligne["erreur"] = "Erreur réseau"
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as erreur:
            ligne["status"] = "invalide"
            ligne["erreur"] = str(erreur)
        ligne["duree_secondes"] = round(time.monotonic() - debut, 3)
        with journal.open("a", encoding="utf-8") as fichier:
            fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            fichier.flush()
            os.fsync(fichier.fileno())
        existantes[case_id] = ligne
        depense += ligne["budget_impute_usd"]
        print(case_id[:22], ligne["status"], f"{ligne['budget_impute_usd']:.4f} USD", flush=True)

    valides = sum(entree["status"] == "valide" for entree in existantes.values())
    print(f"Résultat : {valides}/20 valides ; dépense ou réserve : {depense:.4f} USD")


if __name__ == "__main__":
    main()
