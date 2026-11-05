"""Fonctions communes de lecture et de mise en forme du corpus SFT."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


RACINE = Path(__file__).resolve().parents[1]
BRUT = RACINE / "data/raw"
ANNOTATIONS = RACINE / "data/annotations/corpus"
SEED = 3407

SYSTEME_FR = (
    "Tu es un assistant médical bilingue destiné à un prototype pédagogique. "
    "Pour une présentation de patient, indique le niveau ESI, la priorité et une orientation prudente. "
    "Pour une question médicale, réponds directement et précisément. "
    "Ne remplace jamais l'évaluation d'un professionnel de santé."
)
SYSTEME_EN = (
    "You are a bilingual medical assistant for an educational prototype. "
    "For a patient presentation, give the ESI level, priority, and cautious guidance. "
    "For a medical question, answer directly and precisely. "
    "Never replace assessment by a healthcare professional."
)

PRIORITES = {1: "maximale", 2: "maximale", 3: "modérée", 4: "différée", 5: "différée"}
ORIENTATIONS = {
    1: "Prise en charge vitale immédiate.",
    2: "Évaluation médicale très rapide en raison du risque élevé.",
    3: "Évaluation médicale avec plusieurs ressources probablement nécessaires.",
    4: "Évaluation différée ; une catégorie de ressource est probablement nécessaire.",
    5: "Orientation non urgente ; aucune ressource complémentaire immédiate n'est attendue.",
}
MEDQUAD_QTYPES = {"symptoms", "treatment", "exams and tests", "complications", "prevention"}


def lire_jsonl(chemin):
    """Charge un fichier JSONL en ignorant les lignes vides."""
    return [json.loads(ligne) for ligne in chemin.read_text().splitlines() if ligne.strip()]


def normaliser(texte):
    """Normalise espaces et casse pour comparer deux sollicitations."""
    return " ".join((texte or "").split()).casefold()


def ordre_stable(valeur):
    """Produit un ordre reproductible sans dépendre de l’ordre des fichiers source."""
    return hashlib.sha256(f"{SEED}:{valeur}".encode()).hexdigest()


def exemple(identifiant, systeme, utilisateur, assistant, **metadata):
    """Construit un exemple SFT au format messages avec ses métadonnées."""
    return {
        "id": identifiant,
        "messages": [
            {"role": "system", "content": systeme},
            {"role": "user", "content": utilisateur.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        **metadata,
    }


def justification(annotation):
    """Choisit la justification textuelle la plus informative disponible."""
    texte = annotation.get("justification_classement")
    if texte:
        return texte.strip()
    signes = annotation.get("signes_alerte") or []
    if signes:
        return "; ".join(signe["signe"] for signe in signes)
    return "Le niveau repose sur les éléments explicitement présents dans la présentation."


def charger_triage():
    """Transforme les consensus MediQAl ayant un ESI exact en exemples SFT."""
    cas = {x["case_id"]: x for x in lire_jsonl(ANNOTATIONS / "entrees/cas.jsonl")}
    consensus = lire_jsonl(ANNOTATIONS / "consensus_3_modeles.jsonl")
    lignes = []
    for vote in consensus:
        esi = vote.get("esi")
        if vote.get("statut") != "retenu" or esi is None:
            continue
        annotateurs = [
            annotation for annotation in vote["annotations"].values()
            if annotation.get("esi") == esi
        ]
        riche = next(
            (a for a in annotateurs if a.get("justification_classement")), annotateurs[0]
        )
        reponse = (
            f"Niveau ESI : {esi}\n"
            f"Priorité : urgence {PRIORITES[esi]}\n"
            f"Justification : {justification(riche)}\n"
            f"Orientation : {ORIENTATIONS[esi]}\n"
            "Cet avis d'orientation ne remplace pas une évaluation médicale."
        )
        case_id = vote["case_id"]
        lignes.append(exemple(
            case_id,
            SYSTEME_FR,
            cas[case_id]["texte_source"],
            reponse,
            langue="fr",
            source="ANR-MALADES/MediQAl",
            configuration="mcqu+mcqm",
            source_split="train",
            type_tache="triage",
            licence="CC-BY-4.0",
            niveau_confiance="majorite_3_modeles_esi_exact",
            esi=esi,
            urgence=vote["urgence"],
        ))
    if len(lignes) != 701:
        raise ValueError(f"701 cas de triage attendus, {len(lignes)} obtenus")
    return lignes


def charger_mediqal(nombre):
    """Sélectionne un nombre donné de QCU français sans cas clinique."""
    candidats = []
    chemin = BRUT / "MediQAl/mcqu/train.json"
    for entree in lire_jsonl(chemin):
        if (entree.get("clinical_case") or "").strip():
            continue
        lettre = (entree.get("correct_answers") or "").strip().upper()
        if lettre not in "ABCDE" or len(lettre) != 1:
            continue
        choix = [
            f"{lettre_choix}) {entree.get('answer_' + lettre_choix.lower(), '').strip()}"
            for lettre_choix in "ABCDE"
            if (entree.get("answer_" + lettre_choix.lower()) or "").strip()
        ]
        question = (entree.get("question") or "").strip()
        reponse = (entree.get("answer_" + lettre.lower()) or "").strip()
        if not question or not reponse:
            continue
        utilisateur = question + "\n" + "\n".join(choix)
        candidats.append(exemple(
            "mediqal-mcqu-" + entree["id"],
            SYSTEME_FR,
            utilisateur,
            f"Réponse : {lettre}) {reponse}",
            langue="fr",
            source="ANR-MALADES/MediQAl",
            configuration="mcqu",
            source_split="train",
            type_tache="question_reponse",
            licence="CC-BY-4.0",
            niveau_confiance="reponse_source",
            sujet_medical=entree.get("medical_subject"),
            type_question=entree.get("question_type"),
        ))
    uniques = dedupliquer(candidats)
    uniques.sort(key=lambda x: ordre_stable(x["id"]))
    if len(uniques) < nombre:
        raise ValueError("Réservoir MediQAl insuffisant")
    return uniques[:nombre]


def charger_medquad(nombre):
    """Sélectionne un nombre donné de questions-réponses anglaises exploitables."""
    chemin = BRUT / "MedQuad-MediacalQnADataset/medDataset_processed.csv"
    candidats = []
    with chemin.open(encoding="utf-8-sig", newline="") as fichier:
        for numero, entree in enumerate(csv.DictReader(fichier), 1):
            qtype = (entree.get("qtype") or "").strip().casefold()
            question = (entree.get("Question") or "").strip()
            reponse = (entree.get("Answer") or "").strip()
            if qtype not in MEDQUAD_QTYPES or not question or len(reponse) < 100:
                continue
            identifiant = "medquad-" + hashlib.sha256(question.encode()).hexdigest()
            candidats.append(exemple(
                identifiant,
                SYSTEME_EN,
                question,
                reponse,
                langue="en",
                source="keivalya/MedQuad-MedicalQnADataset",
                configuration="default",
                source_split="non_renseigne",
                type_tache="question_reponse",
                licence="non_renseignee_dans_la_copie_source",
                niveau_confiance="reponse_source",
                qtype=qtype,
                ligne_source=numero,
            ))
    uniques = dedupliquer(candidats)
    uniques.sort(key=lambda x: ordre_stable(x["id"]))
    if len(uniques) < nombre:
        raise ValueError("Réservoir MedQuAD insuffisant")
    return uniques[:nombre]


def dedupliquer(lignes):
    """Conserve la première occurrence de chaque message utilisateur normalisé."""
    vus, sortie = set(), []
    for ligne in lignes:
        cle = normaliser(ligne["messages"][1]["content"])
        if cle and cle not in vus:
            vus.add(cle)
            sortie.append(ligne)
    return sortie


def statistiques(lignes):
    """Résume les langues, tâches, sources et niveaux ESI d’un ensemble."""
    return {
        "nombre": len(lignes),
        "langues": dict(Counter(x["langue"] for x in lignes)),
        "types_tache": dict(Counter(x["type_tache"] for x in lignes)),
        "sources": dict(Counter(x["source"] for x in lignes)),
        "esi_triage": dict(Counter(str(x["esi"]) for x in lignes if x["type_tache"] == "triage")),
    }


def ecrire_jsonl(chemin, lignes):
    """Sérialise une liste d’objets en JSONL UTF-8."""
    chemin.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes))


