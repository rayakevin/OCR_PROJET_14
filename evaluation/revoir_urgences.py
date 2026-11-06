"""Je regroupe les urgences de validation manquées, sans modifier leurs labels."""

from collections import Counter
import json
from pathlib import Path

from evaluation.metrics import extraire_esi, extraire_priorite


def lire_jsonl(fichier):
    return [json.loads(l) for l in Path(fichier).read_text().splitlines() if l.strip()]


def analyser_erreurs(exemples, predictions, reference, historiques):
    """Compare les sorties aux annotations et relie les traductions d'un cas.

    Une erreur signifie ici un désaccord avec la référence du corpus, issue
    d'annotations LLM. Cette revue technique ne valide pas cliniquement le label.
    """
    ids = [x["id"] for x in exemples]
    collections = {"mixte": predictions, "precedent": reference, **historiques}
    for nom, lignes in collections.items():
        if [x["id"] for x in lignes] != ids or len(set(ids)) != len(ids):
            raise ValueError(f"Prédictions {nom} incomplètes ou désordonnées.")
    index = {nom: {x["id"]: x["prediction"] for x in lignes} for nom, lignes in collections.items()}
    erreurs = []
    for exemple in exemples:
        identifiant = exemple["id"]
        prediction = index["mixte"][identifiant]
        if exemple["esi"] > 2 or extraire_priorite(prediction) == "maximale":
            continue
        paire = exemple.get("id_paire", identifiant)
        traductions = [x for x in exemples if x.get("id_paire", x["id"]) == paire]
        erreurs.append({
            "id": identifiant, "id_paire": paire, "langue": exemple["langue"],
            "esi_reference": exemple["esi"], "esi_predit": extraire_esi(prediction),
            "texte": exemple["messages"][1]["content"],
            "cible": exemple["messages"][2]["content"], "prediction": prediction,
            "deja_manquee_par_reference": extraire_priorite(index["precedent"][identifiant]) != "maximale",
            "esi_par_etape": {nom: extraire_esi(index[nom][identifiant]) for nom in historiques},
            "esi_par_langue": {x["langue"]: extraire_esi(index["mixte"][x["id"]]) for x in traductions},
        })
    return {
        "nature": "Revue des désaccords avec les annotations ; sans validation clinique ni ajout au train",
        "nombre_entrees": len(erreurs), "nombre_cas": len({x["id_paire"] for x in erreurs}),
        "par_langue": dict(Counter(x["langue"] for x in erreurs)),
        "deja_manquees_par_reference": sum(x["deja_manquee_par_reference"] for x in erreurs),
        "erreurs": erreurs,
    }
