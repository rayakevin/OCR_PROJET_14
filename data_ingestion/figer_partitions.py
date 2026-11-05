"""Je fige une version nettoyée sans redistribuer les partitions historiques."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))


def lire_jsonl(chemin):
    """Charge les objets JSONL non vides dans l'ordre du fichier."""
    return [json.loads(l) for l in chemin.read_text().splitlines() if l.strip()]


def normaliser(texte):
    """Compare les prompts indépendamment de la casse et des espaces."""
    return ' '.join(texte.casefold().split())


def groupes(lignes):
    """Garde les deux langues d'un cas dans un groupe indivisible."""
    resultat = defaultdict(list)
    for r in lignes:
        resultat[r.get('id_paire', r['id'])].append(r)
    return resultat


def nettoyer(sources, ids_exposes):
    """Privilégie validation, puis test, puis train pour les prompts dupliqués.

    Je retire les groupes entiers en conflit. Le test exposé reste traçable dans
    les exclusions ; il ne devient jamais une nouvelle donnée d'apprentissage.
    Les prompts DPO sont disjoints de tout le SFT historique, même écarté.
    """
    sorties, exclusions, vus = {}, [], set()
    reserves = {normaliser(r['messages'][1]['content']) for nom, lignes in sources.items()
                if nom.startswith('phase') and not nom.endswith('_train') for r in lignes}
    for split in ('validation', 'test', 'train'):
        for phase in ('phase1_qa', 'phase2_triage'):
            nom = f'{phase}_{split}'
            sorties[nom] = []
            for identifiant, lignes in groupes(sources[nom]).items():
                prompts = {normaliser(r['messages'][1]['content']) for r in lignes}
                motif = ('pilote_test_deja_expose' if split == 'test' and
                         any(r['id'] in ids_exposes for r in lignes) else
                         'prompt_commun_autre_groupe' if prompts & (vus | reserves if split == 'train' else vus)
                         else None)
                if motif:
                    exclusions.append({'fichier': nom, 'groupe': identifiant,
                                       'ids': [r['id'] for r in lignes], 'motif': motif})
                else:
                    sorties[nom].extend(lignes)
                    vus.update(prompts)

    prompts_sft = {normaliser(r['messages'][1]['content']) for nom, lignes in sources.items()
                   if nom.startswith('phase') for r in lignes}
    prompts_dpo = set()
    for split in ('validation', 'train'):
        nom = f'dpo_{split}'
        sorties[nom] = []
        for r in sources[nom]:
            prompt = normaliser(r['prompt'][-1]['content'])
            choisi, rejete = (normaliser(r[k][-1]['content']) for k in ('chosen', 'rejected'))
            motif = ('reponse_vide_ou_identique' if not choisi or not rejete or choisi == rejete else
                     'prompt_commun_sft' if prompt in prompts_sft else
                     'prompt_dpo_deja_present' if prompt in prompts_dpo else None)
            if motif:
                exclusions.append({'fichier': nom, 'ids': [r['id']], 'motif': motif})
            else:
                sorties[nom].append(r)
                prompts_dpo.add(prompt)
    return sorties, exclusions


def verifier(sorties):
    """Vérifie identifiants, prompts et intégrité des paires après filtrage."""
    ids_vus, prompts_vus = set(), set()
    for nom, lignes in sorties.items():
        assert len(lignes) == len({r['id'] for r in lignes}), nom
        assert not ids_vus & {r['id'] for r in lignes}, nom
        ids_vus.update(r['id'] for r in lignes)
        prompts = {normaliser(r['messages'][1]['content'] if nom.startswith('phase')
                              else r['prompt'][-1]['content']) for r in lignes}
        assert not prompts_vus & prompts, nom
        prompts_vus.update(prompts)
        if nom.startswith('phase2'):
            assert all(len(g) == 2 and {r['langue'] for r in g} == {'fr', 'en'}
                       for g in groupes(lignes).values()), nom


def ecrire_immuable(chemin, texte):
    """Réexécute à l'identique, mais refuse de modifier une version déjà figée."""
    if chemin.exists() and chemin.read_text() != texte:
        raise ValueError(f'Version existante différente : {chemin}')
    chemin.write_text(texte, encoding='utf-8')


def main():
    """Produit les partitions v3, les exclusions et leurs empreintes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=RACINE / 'data/processed/sft_deux_phases')
    parser.add_argument('--output', type=Path, default=RACINE / 'data/processed/partitions_v3')
    parser.add_argument('--exposition', type=Path, default=RACINE / 'evaluation/pilote_expose.json')
    args = parser.parse_args()
    noms = [f'{p}_{s}' for p in ('phase1_qa', 'phase2_triage') for s in ('train', 'validation', 'test')]
    noms += ['dpo_train', 'dpo_validation']
    sources = {n: lire_jsonl(args.source / f'{n}.jsonl') for n in noms}
    exposition = json.loads(args.exposition.read_text())
    sorties, exclusions = nettoyer(sources, set(exposition['ids']))
    verifier(sorties)
    args.output.mkdir(parents=True, exist_ok=True)
    rapport = {'politique': 'validation > test > train ; exclusion des paires exposées ; DPO disjoint du SFT',
               'exposition_pilote': exposition, 'exclusions': exclusions, 'fichiers': {}}
    for nom, lignes in sorties.items():
        chemin = args.output / f'{nom}.jsonl'
        ecrire_immuable(chemin, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in lignes))
        rapport['fichiers'][nom] = {'avant': len(sources[nom]), 'apres': len(lignes),
            'source_sha256': hashlib.sha256((args.source / f'{nom}.jsonl').read_bytes()).hexdigest(),
            'sha256': hashlib.sha256(chemin.read_bytes()).hexdigest()}
    ecrire_immuable(args.output / 'manifest.json', json.dumps(rapport, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(rapport['fichiers'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
