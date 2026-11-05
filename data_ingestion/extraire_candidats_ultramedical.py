"""Je sélectionne des présentations anglaises candidates au triage dans UltraMedical."""

import hashlib
import json
from pathlib import Path
import re

import ijson


RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "data/raw/UltraMedical-Preference/data/train.json"
SORTIE = RACINE / "data/annotations/ultramedical_triage/entrees"
QUOTAS = {
    "ChatDoctor": 1000,
    "Medical-Instruct-120k": 650,
    "MedInstruct-52k": 350,
}
PRESENTATION = re.compile(
    r"\b(I have|I am|my |patient|woman|man|boy|girl|infant|child|year[- ]old|presents?|reports?|complains?|history of)\b",
    re.IGNORECASE,
)
OPTIONS_QCM = re.compile(r"(?:^|\n)\s*[A-E][.)]\s+", re.MULTILINE)


def ordre(prompt_id):
    """Produit une clé pseudo-aléatoire stable à partir du prompt_id."""
    return hashlib.sha256(("3407:" + prompt_id).encode()).hexdigest()


def admissible(prompt):
    """Repère une présentation de patient exploitable et écarte les QCM explicites."""
    texte = " ".join((prompt or "").split())
    return 100 <= len(texte) <= 3000 and PRESENTATION.search(texte) and not OPTIONS_QCM.search(prompt)


def extraire():
    """Sélectionne 2 000 candidats selon les quotas et écrit leur manifeste."""
    reservoirs = {source: [] for source in QUOTAS}
    comptes = {source: 0 for source in QUOTAS}
    with SOURCE.open("rb") as fichier:
        for entree in ijson.items(fichier, "item"):
            prompt_id = entree.get("prompt_id", "")
            source = prompt_id.split(",", 1)[0]
            if source not in QUOTAS:
                continue
            prompt = (entree.get("prompt") or "").strip()
            if not admissible(prompt):
                continue
            comptes[source] += 1
            reservoirs[source].append({
                "case_id": "ultramedical-" + hashlib.sha256(prompt_id.encode()).hexdigest(),
                "texte_source": prompt,
                "references": [{
                    "source": "TsinghuaC3I/UltraMedical-Preference",
                    "split": "train",
                    "prompt_id": prompt_id,
                    "sous_source": source,
                }],
            })

    selection = []
    for source, quota in QUOTAS.items():
        uniques = {x["references"][0]["prompt_id"]: x for x in reservoirs[source]}
        lignes = sorted(uniques.values(), key=lambda x: ordre(x["references"][0]["prompt_id"]))
        if len(lignes) < quota:
            raise ValueError(f"Réservoir insuffisant pour {source}: {len(lignes)} < {quota}")
        selection.extend(lignes[:quota])
    selection.sort(key=lambda x: x["case_id"])
    if len(selection) != sum(QUOTAS.values()):
        raise ValueError("2 000 candidats attendus")

    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "cas.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selection)
    )
    manifeste = {
        "source": "TsinghuaC3I/UltraMedical-Preference",
        "licence_declaree": "MIT",
        "split_source": "train",
        "seed": 3407,
        "filtres": "présentation patient, 100 à 3000 caractères, hors QCM explicite",
        "candidats_disponibles": comptes,
        "quotas": QUOTAS,
        "nombre_selectionne": len(selection),
        "usage": "candidats au triage SFT; identifiants à exclure du DPO",
    }
    (SORTIE / "manifeste.json").write_text(json.dumps(manifeste, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifeste, ensure_ascii=False, indent=2))
    return manifeste


if __name__ == "__main__":
    extraire()
