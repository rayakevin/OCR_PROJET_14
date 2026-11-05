"""Je prépare un pilote DPO avec le prompt et l'EOS déjà validés en SFT."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from data_ingestion.figer_partitions import ecrire_immuable, lire_jsonl
from training.format_sft import construire_prompt


def formater_paire(exemple, tokenizer, longueur_max, systeme):
    """Conserve les deux réponses entières et refuse une frontière ambiguë.

    Le format texte évite que TRL réapplique un autre chat template. Les réponses
    finissent par l'EOS natif, comme le SFT, et non par la balise im_end.
    """
    if [m['role'] for m in exemple['prompt']] != ['user']:
        raise ValueError('Le pilote attend une question utilisateur unique.')
    prompt = construire_prompt(tokenizer, [{'role': 'system', 'content': systeme}] + exemple['prompt'])
    ids_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    resultat = {'id': exemple['id'], 'prompt': prompt, 'label_type': exemple['label_type']}
    maximum = 0
    for cle in ('chosen', 'rejected'):
        if [m['role'] for m in exemple[cle]] != ['assistant']:
            raise ValueError('Une seule réponse assistant est attendue.')
        texte = exemple[cle][0]['content']
        if not texte.endswith(tokenizer.eos_token):
            texte += tokenizer.eos_token
        ids = tokenizer.encode(prompt + texte, add_special_tokens=False)
        if ids[:len(ids_prompt)] != ids_prompt or ids[-1] != tokenizer.eos_token_id:
            raise ValueError(f"Frontière de tokenisation incorrecte : {exemple['id']}")
        maximum = max(maximum, len(ids))
        resultat[cle] = texte
    if maximum > longueur_max:
        return None, maximum
    return resultat, maximum


def selectionner(lignes, nombre):
    """Échantillonne par catégorie de préférence, sans utiliser les résultats du modèle."""
    volumes = Counter(r['label_type'] for r in lignes)
    ideaux = {c: min(nombre, len(lignes)) * n / len(lignes) for c, n in volumes.items()}
    quotas = {c: int(n) for c, n in ideaux.items()}
    for c in sorted(quotas, key=lambda c: (-(ideaux[c] - quotas[c]), c))[:min(nombre, len(lignes)) - sum(quotas.values())]:
        quotas[c] += 1
    def ordre_stable(identifiant):
        return hashlib.sha256(f'3407:{identifiant}'.encode()).hexdigest()
    selection = []
    for categorie, quota in sorted(quotas.items()):
        candidats = sorted((r for r in lignes if r['label_type'] == categorie), key=lambda r: ordre_stable(r['id']))
        selection.extend(candidats[:quota])
    return sorted(selection, key=lambda r: ordre_stable(r['id']))


def main():
    """Audite les longueurs puis fige les 256/64 paires du pilote."""
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=RACINE / 'data/processed/partitions_v3')
    parser.add_argument('--output', type=Path, default=RACINE / 'data/processed/dpo_pilote_v3')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-length', type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base', revision=args.revision)
    systemes = {r['messages'][0]['content'] for r in lire_jsonl(args.data_dir / 'phase1_qa_train.jsonl')
                if r['langue'] == 'en'}
    assert len(systemes) == 1, 'La consigne anglaise SFT doit être unique.'
    systeme = systemes.pop()
    args.output.mkdir(parents=True, exist_ok=True)
    rapport = {'revision': args.revision, 'max_length': args.max_length, 'seed': 3407, 'splits': {}}
    for split, quota in (('train', 256), ('validation', 64)):
        source = args.data_dir / f'dpo_{split}.jsonl'
        eligibles, longs = [], []
        for r in lire_jsonl(source):
            paire, longueur = formater_paire(r, tokenizer, args.max_length, systeme)
            if paire is None:
                longs.append({'id': r['id'], 'tokens_max': longueur})
            else:
                eligibles.append(paire)
        retenus = selectionner(eligibles, quota)
        if len(retenus) != quota:
            raise ValueError('Pas assez de paires éligibles pour le pilote.')
        cible = args.output / f'{split}.jsonl'
        ecrire_immuable(cible, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in retenus))
        rapport['splits'][split] = {'eligibles': len(eligibles), 'longs_non_tronques': longs,
            'retenus': len(retenus), 'categories': dict(Counter(r['label_type'] for r in retenus)),
            'ids': [r['id'] for r in retenus], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sha256': hashlib.sha256(cible.read_bytes()).hexdigest()}
    ecrire_immuable(args.output / 'manifest.json', json.dumps(rapport, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({s: {k: v for k, v in r.items() if k not in ('ids', 'longs_non_tronques')}
                      for s, r in rapport['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
