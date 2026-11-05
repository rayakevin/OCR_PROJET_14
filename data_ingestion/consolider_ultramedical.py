"""Je consolide les annotations UltraMedical sans forcer un quota de cas."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from annoter_pilote_mistral import valider_vote_compact
from schema_annotation import valider


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/annotations/ultramedical_triage"
MODELES = ("codex", "claude", "mistral")


def lire_jsonl(chemin):
    """Charge les objets d'un fichier JSONL en ignorant les lignes vides."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def charger_lots(dossier):
    """Concatène les lots d'un annotateur dans leur ordre numérique."""
    lignes = []
    for chemin in sorted(dossier.glob("lot-*.jsonl")):
        lignes.extend(lire_jsonl(chemin))
    return lignes


def majorite(valeurs):
    """Retourne la valeur soutenue par au moins deux annotations."""
    if not valeurs:
        return None, 0
    valeur, votes = Counter(valeurs).most_common(1)[0]
    return (valeur, votes) if votes >= 2 else (None, votes)


def calculer_consensus(annotations):
    """Calcule une majorité hiérarchique : qualification, urgence, puis ESI."""
    decision, votes_decision = majorite(
        [annotation["decision_entree"] for annotation in annotations]
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
        return resultat | {"statut": "a_revoir", "motif": "sans_majorite_qualification"}
    if decision != "candidat_initial":
        return resultat | {"statut": "exclu", "motif": "qualification_non_candidate"}

    classes = [
        annotation
        for annotation in annotations
        if annotation["decision_entree"] == "candidat_initial"
        and annotation["statut_classement"] == "classe"
    ]
    urgence, votes_urgence = majorite([annotation["urgence"] for annotation in classes])
    resultat.update(urgence=urgence, votes_urgence=votes_urgence)
    if urgence is None:
        return resultat | {"statut": "a_revoir", "motif": "sans_majorite_urgence"}

    esi, votes_esi = majorite(
        [annotation["esi"] for annotation in classes if annotation["urgence"] == urgence]
    )
    resultat.update(esi=esi, votes_esi=votes_esi)
    if esi is None:
        return resultat | {"statut": "a_revoir", "motif": "sans_majorite_esi"}
    return resultat | {"statut": "retenu", "motif": "majorite_exacte"}


def ordre_reproductible(identifiant, graine=3407):
    """Produit une clé stable pour mélanger les cas sans dépendre de leur ordre source."""
    return hashlib.sha256(f"{graine}:{identifiant}".encode()).hexdigest()


def consolider():
    """Valide les sorties, calcule le consensus et écrit les résultats auditables."""
    cas = lire_jsonl(BASE / "entrees/cas.jsonl")
    ids_attendus = [element["case_id"] for element in cas]
    codex = charger_lots(BASE / "codex")
    claude = charger_lots(BASE / "claude")
    mistral = lire_jsonl(BASE / "mistral/votes.jsonl")
    if [x["case_id"] for x in codex] != ids_attendus:
        raise ValueError("Ordre ou identifiants Codex incorrects")
    if [x["case_id"] for x in claude] != ids_attendus:
        raise ValueError("Ordre ou identifiants Claude incorrects")

    sources = {element["case_id"]: element for element in cas}
    sorties = {
        "codex": {x["case_id"]: x for x in codex},
        "claude": {x["case_id"]: x for x in claude},
        "mistral": {x["case_id"]: x for x in mistral},
    }
    if len(sorties["mistral"]) != len(mistral):
        raise ValueError("Identifiants Mistral dupliqués")

    resultats = []
    for case_id in ids_attendus:
        source = sources[case_id]
        annotations = []
        for modele in ("codex", "claude"):
            annotations.append(valider(sorties[modele][case_id], source))
        if case_id in sorties["mistral"]:
            annotations.append(valider_vote_compact(sorties["mistral"][case_id], source))
        resultat = calculer_consensus(annotations)
        resultats.append({
            "case_id": case_id,
            "texte_source": source["texte_source"],
            "references": source["references"],
            "nombre_annotations": len(annotations),
            **resultat,
        })

    retenus = sorted(
        (x for x in resultats if x["statut"] == "retenu"),
        key=lambda x: ordre_reproductible(x["case_id"]),
    )
    (BASE / "consensus.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in resultats)
    )
    (BASE / "consensus_retenus.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in retenus)
    )
    rapport = {
        "cas_sources": len(cas),
        "annotations": {"codex": len(codex), "claude": len(claude), "mistral_cible": len(mistral)},
        "statuts": dict(Counter(x["statut"] for x in resultats)),
        "motifs": dict(Counter(x["motif"] for x in resultats)),
        "consensus_exacts": len(retenus),
        "objectif_initial": 700,
        "ecart_objectif": max(0, 700 - len(retenus)),
        "esi_retenus": dict(Counter(str(x["esi"]) for x in retenus)),
        "sous_sources_retenues": dict(Counter(
            x["references"][0]["sous_source"] for x in retenus
        )),
        "regle_selection": "majorite exacte sur qualification, urgence et ESI ; aucun quota forcé",
    }
    (BASE / "rapport_consensus.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return rapport


if __name__ == "__main__":
    consolider()
