"""Je partage le même préfixe entre l'apprentissage SFT et la génération."""

FORMAT_SFT = "template_sans_reflexion_eos_natif_v2"


def construire_prompt(tokenizer, messages):
    """Prépare la réponse sans réflexion, selon le template exact du tokenizer."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    blocs = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(blocs) + "\n<|im_start|>assistant\n"


def encoder_exemple(tokenizer, exemple, longueur_max):
    """Masque le prompt et apprend la réponse complète, suivie de l'EOS natif.

    Je refuse un exemple trop long : couper sa réponse enseignerait une fin
    artificielle, et couper le cas pourrait supprimer une information utile.
    Le notebook mesure les longueurs avant de fixer le budget de contexte.
    """
    messages = exemple["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Le corpus attend un dialogue system/user/assistant.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Le tokenizer doit définir un token EOS natif.")
    prompt = tokenizer.encode(construire_prompt(tokenizer, messages[:2]), add_special_tokens=False)
    reponse = tokenizer.encode(messages[2]["content"], add_special_tokens=False)
    reponse += [tokenizer.eos_token_id]
    if len(prompt) + len(reponse) > longueur_max:
        raise ValueError(
            f"{exemple.get('id', 'exemple')} : {len(prompt) + len(reponse)} tokens "
            f"> {longueur_max}. Augmenter --max-length ; aucune troncature silencieuse."
        )
    return {"input_ids": prompt + reponse, "labels": [-100] * len(prompt) + reponse,
            "length": len(prompt) + len(reponse)}
