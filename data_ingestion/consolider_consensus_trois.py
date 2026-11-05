"""Je calcule le consensus majoritaire Codex, Claude et Mistral."""

from collections import Counter
import json
from pathlib import Path

from schema_annotation import valider
from annoter_pilote_mistral import valider_vote_compact


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/annotations/corpus"
MISTRAL = RACINE / "data/annotations/arbitrage_mistral/mistral.jsonl"
SORTIE = BASE / "consensus_3_modeles.jsonl"
RAPPORT = BASE / "rapport_consensus_3_modeles.json"
MODELES = ("codex", "claude", "mistral")


def lire_jsonl(chemin):
    """Charge toutes les lignes non vides d’un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def majorite(valeurs):
    """Retourne une valeur seulement lorsqu’elle reçoit au moins deux votes."""
    compte = Counter(valeurs)
    valeur, votes = compte.most_common(1)[0]
    return (valeur, votes) if votes >= 2 else (None, votes)


def consensus_trois(annotations):
    """Applique successivement la majorité sur la qualification, l’urgence et l’ESI."""
    decision, votes_decision = majorite(
        [annotations[modele]["decision_entree"] for modele in MODELES]
    )
    resultat = {
        "decision_entree": decision,
        "votes_decision": votes_decision,
        "urgence": None,
        "votes_urgence": 0,
        "esi": None,
        "votes_esi": 0,
    }
    if decision is None:
        return resultat | {"statut": "a_revoir", "motif": "aucune_majorite_qualification"}
    if decision != "candidat_initial":
        return resultat | {"statut": "exclu", "motif": "qualification_non_initiale_majoritaire"}

    urgences = [
        annotation["urgence"]
        for annotation in annotations.values()
        if annotation["decision_entree"] == "candidat_initial"
        and annotation["statut_classement"] == "classe"
    ]
    urgence, votes_urgence = majorite(urgences) if urgences else (None, 0)
    resultat["urgence"] = urgence
    resultat["votes_urgence"] = votes_urgence
    if urgence is None:
        return resultat | {"statut": "a_revoir", "motif": "aucune_majorite_urgence"}

    esi_eligibles = [
        annotation["esi"]
        for annotation in annotations.values()
        if annotation["decision_entree"] == "candidat_initial"
        and annotation["statut_classement"] == "classe"
        and annotation["urgence"] == urgence
    ]
    esi, votes_esi = majorite(esi_eligibles)
    resultat["esi"] = esi
    resultat["votes_esi"] = votes_esi
    return resultat | {"statut": "retenu", "motif": "majorite_qualification_et_urgence"}


def charger_annotations(cas):
    """Charge les trois sorties et vérifie leur ordre par rapport au corpus source."""
    attendus = [element["case_id"] for element in cas]
    sorties = {}
    for modele in ("codex", "claude"):
        lignes = []
        for chemin in sorted((BASE / modele).glob("lot-*.jsonl")):
            lignes.extend(lire_jsonl(chemin))
        if [ligne.get("case_id") for ligne in lignes] != attendus:
            raise ValueError(f"Identifiants ou ordre incorrects pour {modele}")
        sorties[modele] = {ligne["case_id"]: ligne for ligne in lignes}

    lignes_mistral = lire_jsonl(MISTRAL)
    if [ligne.get("case_id") for ligne in lignes_mistral] != attendus:
        raise ValueError("Identifiants ou ordre incorrects pour Mistral")
    sorties["mistral"] = {ligne["case_id"]: ligne for ligne in lignes_mistral}
    return sorties


def consolider():
    """Valide les annotations, calcule le consensus et produit son rapport."""
    cas = lire_jsonl(BASE / "entrees/cas.jsonl")
    sorties = charger_annotations(cas)
    resultats = []
    for element in cas:
        case_id = element["case_id"]
        annotations = {modele: sorties[modele][case_id] for modele in MODELES}
        valider(annotations["codex"], element)
        valider(annotations["claude"], element)
        valider_vote_compact(annotations["mistral"], element)
        resultats.append({
            "case_id": case_id,
            "annotations": annotations,
            **consensus_trois(annotations),
        })

    SORTIE.write_text(
        "".join(json.dumps(resultat, ensure_ascii=False) + "\n" for resultat in resultats)
    )
    retenus = [resultat for resultat in resultats if resultat["statut"] == "retenu"]
    rapport = {
        "nombre_cas": len(resultats),
        "modeles": list(MODELES),
        "statuts": dict(Counter(resultat["statut"] for resultat in resultats)),
        "motifs": dict(Counter(resultat["motif"] for resultat in resultats)),
        "decisions_majoritaires": dict(Counter(
            resultat["decision_entree"] or "sans_majorite" for resultat in resultats
        )),
        "urgences_retenues": dict(Counter(resultat["urgence"] for resultat in retenus)),
        "esi_majoritaires_retenus": dict(Counter(
            resultat["esi"] if resultat["esi"] is not None else "sans_majorite"
            for resultat in retenus
        )),
        "accord_unanime_qualification": sum(
            len({resultat["annotations"][modele]["decision_entree"] for modele in MODELES}) == 1
            for resultat in resultats
        ),
        "accord_unanime_urgence_parmi_retenus": sum(
            len({resultat["annotations"][modele]["urgence"] for modele in MODELES}) == 1
            for resultat in retenus
        ),
    }
    RAPPORT.write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return rapport


if __name__ == "__main__":
    consolider()
