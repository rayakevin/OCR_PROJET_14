"""Je crée les jeux train, validation et test des deux phases de SFT."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


RACINE = Path(__file__).resolve().parents[1]
BASE = RACINE / "data/processed/sft_deux_phases"
SEED = 3407
PROPORTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}


def lire_jsonl(chemin):
    """Charge les objets non vides d'un fichier JSONL."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def ecrire_jsonl(chemin, lignes):
    """Écrit une collection d'objets au format JSONL."""
    chemin.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes))


def cle_stable(identifiant):
    """Retourne une clé pseudo-aléatoire reproductible."""
    return hashlib.sha256(f"{SEED}:{identifiant}".encode()).hexdigest()


def quotas_stratifies(strates, total_vise):
    """Répartit un quota exact selon la méthode des plus grands restes."""
    total = sum(strates.values())
    ideaux = {strate: total_vise * nombre / total for strate, nombre in strates.items()}
    quotas = {strate: math.floor(valeur) for strate, valeur in ideaux.items()}
    reste = total_vise - sum(quotas.values())
    ordre = sorted(strates, key=lambda s: (-(ideaux[s] - quotas[s]), str(s)))
    for strate in ordre[:reste]:
        quotas[strate] += 1
    return quotas


def repartir_groupes(groupes, strate):
    """Découpe des groupes indivisibles en respectant les proportions et les strates."""
    par_strate = defaultdict(list)
    for identifiant, exemples in groupes.items():
        par_strate[strate(exemples[0])].append(identifiant)
    for ids in par_strate.values():
        ids.sort(key=cle_stable)

    nombre_groupes = len(groupes)
    cible_validation = round(nombre_groupes * PROPORTIONS["validation"])
    cible_test = round(nombre_groupes * PROPORTIONS["test"])
    volumes = {s: len(ids) for s, ids in par_strate.items()}
    quotas_validation = quotas_stratifies(volumes, cible_validation)
    quotas_test = quotas_stratifies(volumes, cible_test)

    affectation = {}
    for nom_strate, ids in par_strate.items():
        n_val = quotas_validation[nom_strate]
        n_test = quotas_test[nom_strate]
        if n_val + n_test > len(ids):
            raise ValueError("Strate trop petite pour les quotas demandés")
        for identifiant in ids[:n_val]:
            affectation[identifiant] = "validation"
        for identifiant in ids[n_val:n_val + n_test]:
            affectation[identifiant] = "test"
        for identifiant in ids[n_val + n_test:]:
            affectation[identifiant] = "train"

    sorties = {nom: [] for nom in PROPORTIONS}
    for identifiant, exemples in groupes.items():
        sorties[affectation[identifiant]].extend(exemples)
    return sorties


def groupes_individuels(exemples):
    """Traite chaque question-réponse comme un groupe indépendant."""
    return {exemple["id"]: [exemple] for exemple in exemples}


def groupes_bilingues(exemples):
    """Regroupe les versions française et anglaise d'un même cas clinique."""
    groupes = defaultdict(list)
    for exemple in exemples:
        groupes[exemple["id_paire"]].append(exemple)
    if any(len(groupe) != 2 or {x["langue"] for x in groupe} != {"fr", "en"} for groupe in groupes.values()):
        raise ValueError("Chaque cas de triage doit posséder exactement deux langues")
    return dict(groupes)


def resume(lignes):
    """Résume un split pour rendre son contrôle immédiat."""
    return {
        "nombre": len(lignes),
        "langues": dict(Counter(x["langue"] for x in lignes)),
        "sources": dict(Counter(x["source"] for x in lignes)),
        "esi": dict(Counter(str(x["esi"]) for x in lignes if x.get("esi") is not None)),
    }


def construire():
    """Reconstruit séparément les trois splits des phases 1 et 2."""
    maitre_phase1 = BASE / "phase1_qa_corpus.jsonl"
    maitre_phase2 = BASE / "phase2_triage_corpus.jsonl"
    phase1 = (
        lire_jsonl(maitre_phase1) if maitre_phase1.exists()
        else lire_jsonl(BASE / "phase1_qa_train.jsonl") + lire_jsonl(BASE / "phase1_qa_validation.jsonl")
    )
    phase2 = (
        lire_jsonl(maitre_phase2) if maitre_phase2.exists()
        else lire_jsonl(BASE / "phase2_triage_train.jsonl") + lire_jsonl(BASE / "phase2_triage_validation.jsonl")
    )
    if len(phase1) != 2622 or len(phase2) != 2378:
        raise ValueError("Volumes maîtres inattendus")
    ecrire_jsonl(maitre_phase1, phase1)
    ecrire_jsonl(maitre_phase2, phase2)

    splits_phase1 = repartir_groupes(
        groupes_individuels(phase1),
        strate=lambda x: (x["source"], x["langue"]),
    )
    splits_phase2 = repartir_groupes(
        groupes_bilingues(phase2),
        strate=lambda x: x["esi"],
    )
    rapport = {"proportions": PROPORTIONS, "phase1_qa": {}, "phase2_triage": {}}
    for split, lignes in splits_phase1.items():
        ecrire_jsonl(BASE / f"phase1_qa_{split}.jsonl", lignes)
        rapport["phase1_qa"][split] = resume(lignes)
    for split, lignes in splits_phase2.items():
        ecrire_jsonl(BASE / f"phase2_triage_{split}.jsonl", lignes)
        rapport["phase2_triage"][split] = resume(lignes)

    ids_phase2 = {
        split: {x["id_paire"] for x in lignes} for split, lignes in splits_phase2.items()
    }
    if any(ids_phase2[a] & ids_phase2[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Fuite d'une paire bilingue entre deux splits")
    (BASE / "rapport_splits_sft.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
    return rapport


if __name__ == "__main__":
    construire()
