"""Je contrôle la provenance d'une poursuite depuis les poids du SFT mixte."""

import hashlib
import json
from pathlib import Path

from training.format_sft import FORMAT_SFT


def verifier_depart(checkpoint, train, validation, revision):
    """Refuse une autre tâche, un autre corpus ou une époque incomplète.

    La poursuite recharge les poids LoRA ; l'optimiseur et le calendrier sont
    nouveaux. Ce n'est pas une reprise à l'identique d'un entraînement interrompu.
    Les checkpoints et le protocole d'origine restent intacts.
    """
    checkpoint = Path(checkpoint)
    origine = json.loads((checkpoint.parent.parent / "protocole_entrainement.json").read_text())
    etat = json.loads((checkpoint / "trainer_state.json").read_text())
    if origine["arguments"]["phase"] != "sft_mixte":
        raise ValueError("Le checkpoint doit provenir du SFT mixte.")
    if origine["arguments"]["revision"] != revision or origine["format_sft"] != FORMAT_SFT:
        raise ValueError("Socle ou format incompatibles avec le checkpoint.")
    for split, fichier in (("train", train), ("validation", validation)):
        if hashlib.sha256(Path(fichier).read_bytes()).hexdigest() != origine[f"{split}_sha256"]:
            raise ValueError(f"Le corpus {split} diffère de celui du checkpoint.")
    epoch = float(etat["epoch"])
    if epoch <= 0 or abs(epoch - round(epoch)) > 0.01:
        raise ValueError("La poursuite attend la fin d'une époque complète.")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "epoch_initiale": round(epoch),
        "pas_initial": etat["global_step"],
        "adapter_sha256": hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "optimiseur": "reinitialise", "calendrier": "nouveau_cosinus",
    }
