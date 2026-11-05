"""Je prépare les 2 007 cas MediQAl en lots reproductibles pour Codex et Claude."""

import hashlib
import json
from pathlib import Path

from schema_annotation import fingerprint, lire_consigne


RACINE = Path(__file__).resolve().parents[1]
TAILLE_LOT = 50
MODELES = {
    "codex": {"modele": "GPT-6.1 Sol", "reflexion": "moyenne"},
    "claude": {"modele": "Opus 5.5", "reflexion": "moyenne"},
}


def charger_cas():
    """Réunit et déduplique les cas cliniques MediQAl de train."""
    cas_par_texte = {}
    for configuration in ("mcqu", "mcqm"):
        chemin = RACINE / "data/raw/MediQAl" / configuration / "train.json"
        with chemin.open(encoding="utf-8-sig") as fichier:
            for ligne in fichier:
                if not ligne.strip():
                    continue
                entree = json.loads(ligne)
                texte = entree.get("clinical_case") or ""
                cle = " ".join(texte.split())
                if not cle:
                    continue
                cas = cas_par_texte.setdefault(cle, {
                    "case_id": "mediqal-" + hashlib.sha256(cle.encode()).hexdigest(),
                    "texte_source": texte,
                    "references": [],
                })
                cas["references"].append({
                    "source": "ANR-MALADES/MediQAl",
                    "configuration": configuration,
                    "split": "train",
                    "question_id": entree.get("id"),
                })
    return sorted(cas_par_texte.values(), key=lambda cas: cas["case_id"])


def prompt_modele(modele, nombre_lots):
    """Produit la consigne d’exécution locale pour un annotateur donné."""
    dossier = RACINE / "data/annotations/corpus"
    return f"""Modèle : {MODELES[modele]['modele']}. Réflexion : {MODELES[modele]['reflexion']}.

J'annote le corpus MediQAl préparé en {nombre_lots} lots indépendants.
Je lis et applique exactement la consigne commune à chaque cas, sans générer les labels par mots-clés ou par simulation.

Entrées autorisées :
- {dossier / 'entrees/consigne_commune.txt'}
- {dossier / 'entrees/manifeste.json'}
- {dossier / 'entrees/lots'}

Je traite les lots dans l'ordre. Pour chaque lot, je lis tous les textes complets et j'écris un objet JSON par cas, dans le même ordre.
Je n'accède ni aux sorties de l'autre modèle, ni au notebook, ni aux annotations du pilote, ni aux notes manuelles.
Je ne fais aucune recherche web, aucun appel à un autre modèle et aucun appel API.
Je ne modifie ni les textes source, ni la consigne, ni le manifeste.

Je peux utiliser Python pour lire les fichiers, sérialiser mes annotations et appeler la fonction valider de :
{RACINE / 'data_ingestion/schema_annotation.py'}

Après chaque lot, je vérifie : nombre et ordre des identifiants, JSONL valide, citations exactes, cohérence ESI/urgence et abstentions.
Je corrige uniquement ma propre sortie si un contrôle échoue.
Je reprends sans remplacer les lots déjà complets et valides.

Dossier de sortie : {dossier / modele}
Nom attendu : lot-001.jsonl à lot-{nombre_lots:03d}.jsonl.
Un fichier contient exactement les annotations du lot d'entrée portant le même numéro.
Je n'écris aucun autre fichier et ne fais ni commit ni modification du projet.

Dans ma réponse finale, j'indique les lots écrits, le total d'annotations et les contrôles réalisés.
Ces annotations ne constituent ni un consensus ni une validation clinique.
"""


def preparer():
    """Écrit le corpus, ses lots, son manifeste et les prompts des annotateurs."""
    version, consigne = lire_consigne()
    cas = charger_cas()
    if len(cas) != 2007:
        raise ValueError(f"2 007 cas attendus, {len(cas)} obtenus")

    base = RACINE / "data/annotations/corpus"
    entrees = base / "entrees"
    lots = entrees / "lots"
    lots.mkdir(parents=True, exist_ok=True)
    for modele in MODELES:
        (base / modele).mkdir(parents=True, exist_ok=True)

    (entrees / "cas.jsonl").write_text(
        "".join(json.dumps(element, ensure_ascii=False) + "\n" for element in cas),
        encoding="utf-8",
    )
    consigne_enregistree = consigne + "\n"
    (entrees / "consigne_commune.txt").write_text(consigne_enregistree, encoding="utf-8")

    descriptions_lots = []
    for debut in range(0, len(cas), TAILLE_LOT):
        numero = debut // TAILLE_LOT + 1
        contenu = cas[debut:debut + TAILLE_LOT]
        chemin = lots / f"lot-{numero:03d}.jsonl"
        chemin.write_text(
            "".join(json.dumps(element, ensure_ascii=False) + "\n" for element in contenu),
            encoding="utf-8",
        )
        descriptions_lots.append({
            "lot": numero,
            "fichier": chemin.name,
            "nombre": len(contenu),
            "premier_case_id": contenu[0]["case_id"],
            "dernier_case_id": contenu[-1]["case_id"],
            "sha256": hashlib.sha256(chemin.read_bytes()).hexdigest(),
        })

    manifeste = {
        "version_consigne": version,
        "sha256_consigne": hashlib.sha256(consigne_enregistree.encode()).hexdigest(),
        "nombre_cas": len(cas),
        "taille_lot": TAILLE_LOT,
        "nombre_lots": len(descriptions_lots),
        "ordre": "case_id croissant",
        "modeles": MODELES,
        "empreinte_corpus": fingerprint(cas),
        "lots": descriptions_lots,
    }
    (entrees / "manifeste.json").write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for modele in MODELES:
        (entrees / f"prompt_{modele}.txt").write_text(
            prompt_modele(modele, len(descriptions_lots)), encoding="utf-8"
        )

    print(f"{len(cas)} cas ; {len(descriptions_lots)} lots ; {TAILLE_LOT} cas maximum par lot")
    print("Entrées :", entrees)
    print("Sorties attendues :", ", ".join(str(base / modele) for modele in MODELES))
    return manifeste


if __name__ == "__main__":
    preparer()
