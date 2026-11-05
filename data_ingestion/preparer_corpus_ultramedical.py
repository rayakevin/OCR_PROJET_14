"""Je prépare les 2 000 candidats UltraMedical en lots indépendants."""

import hashlib
import json
from pathlib import Path

from schema_annotation import fingerprint, lire_consigne


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/annotations/ultramedical_triage"
TAILLE_LOT = 50
MODELES = {
    "codex": {"modele": "GPT-6.1 Sol", "reflexion": "moyenne"},
    "claude": {"modele": "Opus 5.5", "reflexion": "moyenne"},
}


def lire_jsonl(chemin):
    """Charge un fichier JSONL en ignorant les lignes vides."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def prompt(modele, nombre_lots):
    """Produit la consigne d’annotation UltraMedical pour un modèle donné."""
    configuration = MODELES[modele]
    return f"""Modèle : {configuration['modele']}. Réflexion : {configuration['reflexion']}.

J'annote 2 000 présentations médicales anglaises candidates au triage, réparties en {nombre_lots} lots.
Je lis chaque texte intégralement et j'applique exactement la consigne commune.
Je ne déduis aucune information absente et je distingue l'état initial des résultats ou événements ultérieurs.

Entrées autorisées :
- {BASE / 'entrees/consigne_commune.txt'}
- {BASE / 'entrees/manifeste_annotation.json'}
- {BASE / 'entrees/lots'}

Je traite les lots dans l'ordre. Pour chaque lot, j'écris un objet JSON par cas dans le même ordre.
Je n'accède ni aux sorties de l'autre modèle, ni aux annotations MediQAl, ni au consensus existant.
Je ne fais aucune recherche web, aucun appel à un autre modèle et aucun appel API.
Je ne traduis pas les textes et je conserve les citations en anglais exactement comme dans la source.

Je peux utiliser Python pour lire les fichiers, sérialiser mes annotations et appeler la fonction valider de :
{RACINE / 'data_ingestion/schema_annotation.py'}

Après chaque lot, je vérifie le nombre et l'ordre des identifiants, le JSONL, les citations exactes,
la cohérence ESI/urgence et les abstentions. Je reprends sans remplacer les lots déjà complets et valides.

Dossier de sortie : {BASE / modele}
Fichiers attendus : lot-001.jsonl à lot-{nombre_lots:03d}.jsonl.
Je n'écris aucun autre fichier et ne modifie pas les sources.

Dans ma réponse finale, j'indique les lots écrits, le total d'annotations et les contrôles réalisés.
Ces annotations ne constituent ni un consensus ni une validation clinique.
"""


def preparer():
    """Vérifie les 2 000 candidats puis crée 40 lots reproductibles."""
    cas = lire_jsonl(BASE / "entrees/cas.jsonl")
    if len(cas) != 2000 or len({x["case_id"] for x in cas}) != 2000:
        raise ValueError("2 000 candidats uniques attendus")
    version, consigne = lire_consigne()
    lots = BASE / "entrees/lots"
    lots.mkdir(parents=True, exist_ok=True)
    for modele in MODELES:
        (BASE / modele).mkdir(parents=True, exist_ok=True)

    descriptions = []
    for debut in range(0, len(cas), TAILLE_LOT):
        numero = debut // TAILLE_LOT + 1
        contenu = cas[debut:debut + TAILLE_LOT]
        chemin = lots / f"lot-{numero:03d}.jsonl"
        chemin.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in contenu))
        descriptions.append({
            "lot": numero,
            "fichier": chemin.name,
            "nombre": len(contenu),
            "sha256": hashlib.sha256(chemin.read_bytes()).hexdigest(),
        })

    (BASE / "entrees/consigne_commune.txt").write_text(consigne + "\n")
    manifeste = {
        "version_consigne": version,
        "nombre_cas": len(cas),
        "taille_lot": TAILLE_LOT,
        "nombre_lots": len(descriptions),
        "ordre": "case_id croissant",
        "modeles": MODELES,
        "empreinte_corpus": fingerprint(cas),
        "lots": descriptions,
    }
    (BASE / "entrees/manifeste_annotation.json").write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=2) + "\n"
    )
    for modele in MODELES:
        (BASE / "entrees" / f"prompt_{modele}.txt").write_text(prompt(modele, len(descriptions)))
    print(f"{len(cas)} cas ; {len(descriptions)} lots ; {TAILLE_LOT} cas maximum")
    print("Prompts :", BASE / "entrees/prompt_codex.txt", "et", BASE / "entrees/prompt_claude.txt")
    return manifeste


if __name__ == "__main__":
    preparer()
