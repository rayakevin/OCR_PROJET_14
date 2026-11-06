"""Je transforme les erreurs du SFT sur le train en préférences de triage."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))


def sha256(path):
    """Identifie exactement une entrée de l'expérience."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lire(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def decision(texte):
    """Lit une décision unique, avec les variantes lexicales déjà présentes en SFT.

    Cette lecture du corpus ne change pas le contrat plus strict de l'API.
    Je refuse toute priorité explicitement incompatible avec l'ESI annoncé.
    """
    niveaux = re.findall(r'^\s*(?:Niveau[ _]ESI|ESI level)\s*:\s*([1-5])\s*$', texte, re.M | re.I)
    priorites = re.findall(r'^\s*(?:Priorité|Priority)\s*:\s*(.+?)\s*$', texte, re.M | re.I)
    noms = {'maximale': ('urgence maximale', 'maximum urgency', 'highest urgency', 'maximum emergency'),
            'moderee': ('urgence modérée', 'moderate urgency', 'moderate emergency'),
            'differee': ('urgence différée', 'delayed urgency', 'delayed emergency', 'non-urgent')}
    if len(niveaux) != 1 or len(priorites) != 1:
        return None
    esi = int(niveaux[0])
    priorite = 'maximale' if esi <= 2 else 'moderee' if esi == 3 else 'differee'
    return esi if priorites[0].casefold() in noms[priorite] else None


def miner(exemples, predictions, revision, source_sha, adapter_sha):
    """Garde uniquement les erreurs ESI cohérentes et complètes du train.

    La réponse préférée reste celle du consensus silver. La réponse rejetée est
    réellement produite par le SFT : je n'invente ni symptôme ni contre-exemple.
    Les sorties invalides sont comptées à part, car cet essai cible le classement.
    """
    ids = [x['id'] for x in exemples]
    if len(set(ids)) != len(ids) or [x['id'] for x in predictions] != ids:
        raise ValueError('Prédictions incomplètes, dupliquées ou désordonnées.')
    retenues, exclusions = [], Counter()
    for exemple, prediction in zip(exemples, predictions):
        for cle, valeur in {'split': 'train', 'phase': 'triage', 'revision': revision,
                            'donnees_sha256': source_sha, 'adaptateur_sha256': adapter_sha}.items():
            if prediction.get(cle) != valeur:
                raise ValueError(f'Provenance incorrecte : {cle}, {exemple["id"]}')
        if [m['role'] for m in exemple['messages']] != ['system', 'user', 'assistant']:
            raise ValueError('Trois messages SFT attendus.')
        cible = exemple['messages'][2]['content']
        reference = decision(cible)
        priorite_reference = 'maximale' if reference in (1, 2) else 'moderee' if reference == 3 else 'differee'
        if reference != exemple['esi'] or priorite_reference != exemple['urgence']:
            raise ValueError('Référence silver incohérente.')
        sortie = decision(prediction['prediction']) if prediction['arret'] == 'eos' else None
        if sortie is None:
            exclusions['sortie_invalide_ou_tronquee'] += 1
            continue
        attendu, obtenu = exemple['esi'], sortie
        if attendu == obtenu:
            exclusions['esi_correct'] += 1
            continue
        categorie = ('sous_triage_critique' if attendu <= 2 < obtenu else
                     'sous_triage' if obtenu > attendu else 'sur_triage')
        retenues.append({'id': exemple['id'], 'id_paire': exemple['id_paire'],
            'langue': exemple['langue'], 'label_type': categorie,
            'systeme': exemple['messages'][0]['content'],
            'prompt': [exemple['messages'][1]],
            'chosen': [{'role': 'assistant', 'content': cible}],
            'rejected': [{'role': 'assistant', 'content': prediction['prediction']}],
            'esi_reference': attendu, 'esi_rejete': obtenu})
    return retenues, dict(exclusions)


def repartir(paires, seed=3407):
    """Isole 20 % des cas avant DPO, en gardant FR/EN dans le même groupe.

    Ce contrôle interne n'est inédit que pour le DPO : le SFT a déjà vu ces cas.
    L'évaluation métier utilise ensuite la validation v3, jamais ces scores.
    """
    groupes = sorted({x['id_paire'] for x in paires})
    if len(groupes) < 10:
        raise ValueError('Moins de dix cas erronés : pilote insuffisant, pas de remplissage artificiel.')
    random.Random(seed).shuffle(groupes)
    controle = set(groupes[:max(2, round(len(groupes) * .2))])
    splits = {s: [x for x in paires if (x['id_paire'] in controle) == (s == 'validation')]
              for s in ('train', 'validation')}
    directions = {x['label_type'] for x in splits['train']}
    if 'sur_triage' not in directions or 'sous_triage_critique' not in directions:
        raise ValueError('Le train DPO doit contenir sous-triages critiques ET sur-triages.')
    # Pas de duplication pour imposer un équilibre artificiel : je mesure la distribution.
    return splits


def main():
    """Vérifie la provenance, refuse toute troncature et fige les paires retenues."""
    from transformers import AutoTokenizer
    from training.preparer_dpo import formater_paire
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.source.name != 'phase2_triage_train.jsonl':
        raise ValueError('Seul le train triage est autorisé.')
    paires, exclusions = miner(lire(args.source), lire(args.predictions), args.revision,
                              sha256(args.source), sha256(args.adapter / 'adapter_model.safetensors'))
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base', revision=args.revision)
    eligibles = []
    for paire in paires:
        formattee, longueur = formater_paire(paire, tokenizer, 4096, paire['systeme'])
        if formattee is None:
            exclusions['trop_long'] = exclusions.get('trop_long', 0) + 1
        else:
            eligibles.append(formattee | {k: paire[k] for k in
                ('id_paire', 'langue', 'esi_reference', 'esi_rejete')})
    splits = repartir(eligibles)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Dossier DPO non vide : aucune donnée écrasée.')
    manifeste = {'revision': args.revision, 'max_length': 4096, 'seed': 3407,
        'type': 'erreurs_reelles_du_sft_sur_train_triage_silver',
        'source_sha256': sha256(args.source), 'predictions_sha256': sha256(args.predictions),
        'adapter_sha256': sha256(args.adapter / 'adapter_model.safetensors'),
        'exclusions': exclusions, 'splits': {}, 'test_lu': False}
    for split, lignes in splits.items():
        cible = args.output / f'{split}.jsonl'
        cible.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in lignes))
        manifeste['splits'][split] = {'retenus': len(lignes), 'sha256': sha256(cible),
            'cas': len({x['id_paire'] for x in lignes}),
            'categories': dict(Counter(x['label_type'] for x in lignes)),
            'langues': dict(Counter(x['langue'] for x in lignes))}
    (args.output / 'manifest.json').write_text(json.dumps(manifeste, ensure_ascii=False, indent=2))
    print(json.dumps(manifeste, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
