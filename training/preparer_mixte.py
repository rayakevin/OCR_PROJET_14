"""Je réunis QA et triage sans changer leurs partitions ni lire le test."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata


def construire_mixte(donnees, sortie, seed=3407):
    """Mélange chaque split séparément ; refuse les chevauchements train/validation.

    Chaque exemple apparaît une fois : il n'y a ni sur-échantillonnage ni
    pondération de loss par tâche. La proportion de tokens sera mesurée dans
    le notebook, car une proportion d'exemples n'est pas un poids de gradient.
    """
    donnees, sortie = Path(donnees), Path(sortie)
    splits, sources, statistiques = {}, {}, {}
    for split in ('train', 'validation'):
        lignes = []
        for phase, prefixe in (('qa', 'phase1_qa'), ('triage', 'phase2_triage')):
            fichier = donnees / f'{prefixe}_{split}.jsonl'
            exemples = [json.loads(l) for l in fichier.read_text().splitlines() if l.strip()]
            sources[fichier.name] = hashlib.sha256(fichier.read_bytes()).hexdigest()
            lignes.extend({**x, 'phase_sft': phase} for x in exemples)
        if len({x['id'] for x in lignes}) != len(lignes):
            raise ValueError(f'Identifiants dupliqués dans {split}.')
        if any([m['role'] for m in x['messages']] != ['system', 'user', 'assistant'] for x in lignes):
            raise ValueError('Format SFT inattendu.')
        random.Random(seed).shuffle(lignes)
        splits[split] = lignes
        statistiques[split] = {
            'nombre': len(lignes),
            'par_tache': dict(Counter(x['phase_sft'] for x in lignes)),
            'par_langue': dict(Counter(x['langue'] for x in lignes)),
        }
    for nom, cle in (
        ('identifiant', lambda x: x['id']),
        ('groupe bilingue', lambda x: x.get('id_paire', x['id'])),
        ('prompt', lambda x: ' '.join(unicodedata.normalize(
            'NFKC', x['messages'][1]['content']).casefold().split())),
    ):
        if {cle(x) for x in splits['train']} & {cle(x) for x in splits['validation']}:
            raise ValueError(f'Chevauchement train/validation : {nom}.')
    sortie.mkdir(parents=True, exist_ok=True)
    if any(sortie.iterdir()):
        raise ValueError('Le dossier du mélange doit être vide.')
    for split, lignes in splits.items():
        (sortie / f'{split}.jsonl').write_text(
            ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in lignes), encoding='utf-8')
    manifeste = {'seed': seed, 'politique': 'Union sans répétition ; partitions v3 conservées ; test non lu',
                 'sources_sha256': sources, 'splits': statistiques}
    (sortie / 'manifest.json').write_text(json.dumps(manifeste, ensure_ascii=False, indent=2) + '\n')
    return manifeste
