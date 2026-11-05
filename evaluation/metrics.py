"""Métriques reproductibles pour la QA médicale et le triage."""

from collections import Counter
import re

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


PRIORITES = ("maximale", "moderee", "differee")
ESI_VERS_PRIORITE = {1: "maximale", 2: "maximale", 3: "moderee", 4: "differee", 5: "differee"}


def extraire_esi(texte):
    """Extrait un niveau ESI explicite compris entre 1 et 5."""
    # Les modèles insèrent parfois du Markdown entre le libellé et la valeur.
    motif = r"(?:niveau\s+)?ESI(?:\s+level|\s+niveau)?[*_\s]*[:=-]?[*_\s]*([1-5])\b"
    trouve = re.search(motif, texte or "", re.IGNORECASE)
    return int(trouve.group(1)) if trouve else None


def extraire_priorite(texte):
    """Normalise les formulations françaises et anglaises des trois priorités."""
    valeur = (texte or "").casefold()
    motifs = {
        "maximale": ("urgence maximale", "maximum urgency", "highest urgency", "immediate urgency"),
        "moderee": ("urgence modérée", "urgence moderee", "moderate urgency"),
        "differee": ("urgence différée", "urgence differee", "delayed urgency", "deferred urgency"),
    }
    for priorite, formulations in motifs.items():
        if any(formulation in valeur for formulation in formulations):
            return priorite
    esi = extraire_esi(texte)
    return ESI_VERS_PRIORITE.get(esi)


def intervalle_wilson(succes, total, z=1.96):
    """Calcule l'intervalle de confiance de Wilson à 95 % d'une proportion."""
    if total == 0:
        return None
    proportion = succes / total
    denominateur = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    marge = z * ((proportion * (1 - proportion) / total + z * z / (4 * total**2)) ** 0.5)
    return [round((centre - marge) / denominateur, 4), round((centre + marge) / denominateur, 4)]


def _rapport_classes(attendus, predictions, classes):
    """Produit accuracy, macro-F1, rappel et matrice de confusion sur des classes fixes."""
    pred_scoring = [prediction if prediction in classes else "invalide" for prediction in predictions]
    rappel = recall_score(attendus, pred_scoring, labels=list(classes), average=None, zero_division=0)
    supports = Counter(attendus)
    corrects = Counter(a for a, p in zip(attendus, pred_scoring) if a == p)
    return {
        "accuracy": round(accuracy_score(attendus, pred_scoring), 4),
        "macro_f1": round(f1_score(attendus, pred_scoring, labels=list(classes), average="macro", zero_division=0), 4),
        "rappel_par_classe": {classe: round(float(score), 4) for classe, score in zip(classes, rappel)},
        "support_par_classe": {classe: supports[classe] for classe in classes},
        "ic95_rappel": {
            classe: intervalle_wilson(corrects[classe], supports[classe]) for classe in classes
        },
        "matrice_confusion": confusion_matrix(
            attendus, pred_scoring, labels=list(classes) + ["invalide"]
        ).tolist(),
        "ordre_matrice": list(classes) + ["invalide"],
    }


def rapport_triage(exemples, predictions):
    """Évalue l'ESI, la priorité, le sous-triage et le comportement de sortie."""
    par_id = {ligne["id"]: ligne["prediction"] for ligne in predictions}
    lignes = [(exemple, par_id.get(exemple["id"], "")) for exemple in exemples]
    esi_attendus = [x[0]["esi"] for x in lignes]
    esi_predits = [extraire_esi(x[1]) for x in lignes]
    priorites_attendues = [ESI_VERS_PRIORITE[x] for x in esi_attendus]
    priorites_predites = [extraire_priorite(x[1]) for x in lignes]
    valides = [(a, p) for a, p in zip(esi_attendus, esi_predits) if p is not None]
    sous_triages = sum(p > a for a, p in valides)
    sur_triages = sum(p < a for a, p in valides)
    critiques = sum(a <= 2 and p >= 3 for a, p in valides)
    # Les deux corpus français emploient respectivement le singulier et le pluriel.
    # Ce marqueur vérifie la présence d'une formule, pas la prudence clinique du texte.
    disclaimer = ("ne remplace pas", "ne remplacent pas", "does not replace", "do not replace")
    paires = {}
    for (exemple, _), prediction in zip(lignes, esi_predits):
        if exemple.get("id_paire"):
            paires.setdefault(exemple["id_paire"], []).append(prediction)
    paires_completes = [valeurs for valeurs in paires.values() if len(valeurs) == 2]
    paires_valides = [valeurs for valeurs in paires_completes if None not in valeurs]
    rapport = {
        "nombre": len(lignes),
        "esi": _rapport_classes(
            [str(x) for x in esi_attendus],
            [str(x) if x is not None else None for x in esi_predits],
            ("1", "2", "3", "4", "5"),
        ),
        "priorite": _rapport_classes(priorites_attendues, priorites_predites, PRIORITES),
        "taux_sortie_esi_valide": round(len(valides) / len(lignes), 4),
        "erreur_absolue_moyenne_esi": round(sum(abs(a - p) for a, p in valides) / len(valides), 4) if valides else None,
        "taux_sous_triage": round(sous_triages / len(valides), 4) if valides else None,
        "taux_sous_triage_critique": round(critiques / len(valides), 4) if valides else None,
        "taux_sur_triage": round(sur_triages / len(valides), 4) if valides else None,
        "taux_avertissement": round(sum(any(m in texte.casefold() for m in disclaimer) for _, texte in lignes) / len(lignes), 4),
        "coherence_bilingue_esi": {
            "paires": len(paires_completes),
            "paires_analysables": len(paires_valides),
            "taux_meme_prediction": round(
                sum(a == b for a, b in paires_valides) / len(paires_valides), 4
            ) if paires_valides else None,
        },
    }
    rapport["par_langue"] = {
        langue: rapport_triage(
            [exemple for exemple, _ in lignes if exemple["langue"] == langue], predictions
        ) | {"par_langue": None}
        for langue in ("fr", "en")
    } if {x[0]["langue"] for x in lignes} == {"fr", "en"} else None
    return rapport


def extraire_lettre_qcm(texte):
    """Extrait une réponse A-E lorsqu'elle est annoncée explicitement."""
    trouve = re.search(r"(?:réponse|answer|option|choix)\s*[:=-]?\s*([A-E])\b", texte or "", re.I)
    return trouve.group(1).upper() if trouve else None


def _tokens(texte):
    return re.findall(r"\w+", (texte or "").casefold(), re.UNICODE)


def f1_lexical(reference, prediction):
    """Calcule un F1 lexical simple pour les réponses ouvertes."""
    ref, pred = Counter(_tokens(reference)), Counter(_tokens(prediction))
    communs = sum((ref & pred).values())
    if not ref or not pred or communs == 0:
        return 0.0
    precision, rappel = communs / sum(pred.values()), communs / sum(ref.values())
    return 2 * precision * rappel / (precision + rappel)


def rapport_qa(exemples, predictions):
    """Évalue séparément les QCM français et les réponses ouvertes anglaises."""
    par_id = {ligne["id"]: ligne["prediction"] for ligne in predictions}
    qcm, ouvertes = [], []
    for exemple in exemples:
        prediction = par_id.get(exemple["id"], "")
        reference = exemple["messages"][2]["content"]
        lettre = extraire_lettre_qcm(reference)
        if lettre:
            qcm.append((lettre, extraire_lettre_qcm(prediction)))
        else:
            ouvertes.append(f1_lexical(reference, prediction))
    return {
        "nombre": len(exemples),
        "qcm": {
            "nombre": len(qcm),
            "accuracy": round(sum(a == p for a, p in qcm) / len(qcm), 4) if qcm else None,
            "taux_reponse_analysable": round(sum(p is not None for _, p in qcm) / len(qcm), 4) if qcm else None,
        },
        "reponses_ouvertes": {
            "nombre": len(ouvertes),
            "f1_lexical_moyen": round(sum(ouvertes) / len(ouvertes), 4) if ouvertes else None,
        },
    }
