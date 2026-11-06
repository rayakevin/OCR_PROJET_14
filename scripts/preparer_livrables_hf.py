"""Prépare les dépôts HF de remise, avec une liste fermée et des empreintes.

Les dépôts restent privés : je ne transforme pas une licence absente en droit de
redistribution. L'accès de l'évaluateur se règle séparément, sans publier le corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RACINE = Path(__file__).resolve().parents[1]
DPO = RACINE / 'data/colab_archive/dpo_triage_cible_20261106_132004'


def copier(source, cible):
    cible.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, cible)


def preparer(destination):
    """Copie les partitions v3 réelles, sans rééchantillonnage ni réécriture."""
    destination.mkdir(parents=True, exist_ok=False)
    dataset, modele = destination / 'dataset', destination / 'modele'
    configs, origines = [], {}
    for nom, prefixe, source, splits in [
        ('qa', 'phase1_qa_', RACINE/'data/processed/partitions_v3', ['train','validation','test']),
        ('triage', 'phase2_triage_', RACINE/'data/processed/partitions_v3', ['train','validation','test']),
        ('dpo_general', 'dpo_', RACINE/'data/processed/partitions_v3', ['train','validation']),
        ('dpo_cible', '', DPO/'paires', ['train','validation']),
    ]:
        configs.append(f'- config_name: {nom}\n  data_files:')
        exemples = {split: [json.loads(l) for l in (source/f'{prefixe}{split}.jsonl').read_text().splitlines()
                            if l.strip()] for split in splits}
        # HF Datasets exige les mêmes colonnes optionnelles dans tous les lots.
        # J'ajoute seulement null aux métadonnées absentes ; aucun texte ne change.
        colonnes = sorted({k for lignes in exemples.values() for x in lignes for k in x})
        for split in splits:
            original = source/f'{prefixe}{split}.jsonl'
            cible = dataset/nom/f'{split}.jsonl'
            cible.parent.mkdir(parents=True, exist_ok=True)
            cible.write_text(''.join(json.dumps({k: x.get(k) for k in colonnes}, ensure_ascii=False)+'\n'
                                     for x in exemples[split]))
            origines[f'{nom}/{split}.jsonl'] = {'source': str(original.relative_to(RACINE)),
                'sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'nombre': len(exemples[split])}
            configs.append(f'  - split: {split}\n    path: {nom}/{split}.jsonl')
    copier(RACINE/'data/processed/partitions_v3/manifest.json',dataset/'provenance/partitions_v3.json')
    copier(DPO/'paires/manifest.json',dataset/'provenance/dpo_cible.json')
    for nom, dossier in [('MediQAl','MediQAl'), ('MedQuAD','MedQuad-MediacalQnADataset'),
                          ('UltraMedical-Preference','UltraMedical-Preference')]:
        copier(RACINE/'data/raw'/dossier/'README.md', dataset/'provenance'/f'{nom}_source_README.md')
    (dataset/'provenance/fichiers_origine.json').write_text(json.dumps(origines,ensure_ascii=False,indent=2)+'\n')
    carte = '''---
language:
- fr
- en
pretty_name: OCR14 — corpus médical bilingue v3
task_categories:
- text-generation
configs:
CONFIGS
---
# Corpus médical bilingue OCR14 · v3

Je dépose les partitions exactes utilisées dans le projet, pour une remise pédagogique privée.
Le code de construction est dans le dépôt privé `rayakevin/OCR_PROJET_14`, notebook 01.

| Configuration | Train | Validation | Test | Usage |
|---|---:|---:|---:|---|
| qa | 2 098 | 262 | 262 | SFT, connaissances médicales |
| triage | 1 888 | 238 | 226 | SFT, classification ESI / priorité |
| dpo_general | 4 470 | 494 | — | Préférences natives ; pilote 256 / 64 |
| dpo_cible | 355 | 85 | — | Erreurs réelles du SFT sur le train triage |

Le SFT contient 4 974 entrées, dont 3 986 de train. Les 500 entrées de validation
et 488 de test ne calculent pas les gradients. Français et anglais représentent
chacun la moitié des entrées SFT. Les traductions d'un même cas sont groupées.
Le contrôle interne DPO ciblé vient du train SFT ; il n'est pas un test indépendant.

## Schéma et chargement

Les exemples SFT contiennent `messages` (system/user/assistant), un identifiant,
la langue et la provenance. Le triage porte aussi `id_paire`, `esi`, `urgence`.
Les exemples DPO contiennent `prompt`, `chosen`, `rejected` et la provenance.
Pour charger avec HF Datasets, j'uniformise les colonnes optionnelles par
configuration (valeur null si absente). Identifiants, textes, ordre et partitions
restent inchangés ; les empreintes des fichiers originaux sont conservées.

```python
from datasets import load_dataset
corpus = load_dataset("rayakevin/ocr14-medical-bilingue-v3", "triage", token=True,
                      revision="REVISION_INSCRITE_DANS_LA_PREUVE_DE_PUBLICATION")
```

## Provenance et limites

- [ANR-MALADES/MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl)
  (Adrien Bazoge, français) : [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  constatée ; QA reformattée, cas sélectionnés, annotés puis traduits pour le triage.
- MedQuAD, distribution `keivalya/MedQuad-MedicalQnADataset` : anglais, QA ;
  licence non renseignée dans la distribution explorée.
- `TsinghuaC3I/UltraMedical-Preference` : MIT constatée ; anglais, préférences
  et cas dérivés pour le triage.
- FrenchMedMCQA : exploré, non inclus dans ces partitions.

Je n'attribue pas de licence globale à ce mélange. Le dépôt privé ne résout pas
les droits manquants ; toute redistribution au-delà de la remise demande leur
clarification. Les fichiers de provenance identifient les transformations et exclusions.
Les sources initiales sont conservées séparément dans `data/raw`, non redistribuées ici.

Le triage est annoté par consensus de LLM (Codex, Claude, Mistral), puis traduit.
Ce sont des labels silver : ils peuvent partager des biais et ne constituent pas
une vérité clinique. Les données ne permettent pas de certifier un outil médical.
L'audit automatique d'identifiants ne remplace pas une revue humaine.
Le test historique a déjà été observé ; les variantes mixtes/DPO ciblé sont
comparées sur validation et demandent une confirmation indépendante.
'''.replace('CONFIGS','\n'.join(configs))
    (dataset/'README.md').write_text(carte)
    sources = [('', DPO/'source/adapter'),
               ('sft_successif_v3', RACINE/'models/sft_triage_v3/adapter'),
               ('dpo_cible_pas16',DPO/'entrainement/checkpoints/checkpoint-16'),
               ('dpo_cible_pas32',DPO/'entrainement/checkpoints/checkpoint-32')]
    for nom, source in sources:
        for fichier in ('adapter_config.json','adapter_model.safetensors'):
            copier(source/fichier,modele/nom/fichier)
    copier(RACINE/'serving/modele.json',modele/'sft_successif_v3/provenance.json')
    copier(DPO/'comparaison_validation.json',modele/'resultats_dpo_cible_validation.json')
    copier(RACINE/'serving/space/support/resultats.json',modele/'resultats_agreges.json')
    (modele/'README.md').write_text('''---
base_model: Qwen/Qwen3-1.7B-Base
library_name: peft
language:
- fr
- en
pipeline_tag: text-generation
---
# OCR14 · adaptateurs LoRA et comparaison

Je livre le SFT mixte époque 3 à la racine (référence expérimentale équilibrée),
le SFT successif v3 déployé dans son sous-dossier, et les deux checkpoints DPO
ciblés comme résultats expérimentaux **non retenus**. Aucun de ces modèles
n'est validé pour un usage clinique. La référence est un consensus de LLM.

Socle figé : `Qwen/Qwen3-1.7B-Base`, révision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. LoRA r=16, alpha=32,
17 432 576 paramètres entraînables. Le tokenizer et le format doivent être
chargés depuis cette révision avec `training/format_sft.py` du dépôt OCR14.
EOS natif 151643, `enable_thinking=False`. Ces poids sont des adaptateurs,
pas les poids complets du socle. La configuration PEFT précise les modules ciblés.

SFT mixte : trois époques sur 3 986 exemples QA + triage, BF16, learning rate
0,0001, warmup 20, batch effectif 16. SHA-256 attendu de l'adaptateur à la racine :
`62d4e5085fd4dd457c5f7b8bbcd0da944ffaf347b8081123f3247dc342db2877`.

Validation du run DPO : rappel maximal 91 % (SFT), 87 % (DPO16), 83 % (DPO32).
Macro-F1 : 0,7875 → 0,7457 → 0,7373. Le DPO ne satisfait pas les critères
fixés avant lancement. Les rapports fournis donnent QA, sur-triage et cohérence
bilingue. Les paires FR/EN sont corrélées ; les scores ne sont pas cliniques.

Le SFT successif conserve un meilleur rappel urgent sur validation mais oublie
largement la QA. Il reste le modèle du Space actuel ; ce dépôt ne modifie pas
le déploiement. Le test historique du successif est distinct de ces comparaisons.

Je conserve ce dépôt privé compte tenu des restrictions des corpus dérivés.
La licence du socle et celles des données restent distinctes ; je n'attribue pas
une licence globale nouvelle. Code, notebooks et rapport : `rayakevin/OCR_PROJET_14`.
''')
    for dossier in (dataset,modele):
        hashes={str(p.relative_to(dossier)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(dossier.rglob('*')) if p.is_file()}
        (dossier/'empreintes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    return dataset,modele


def publier(dossiers):
    """Versionne sur le compte de remise, sans rendre publics les corpus ou poids."""
    from huggingface_hub import HfApi
    api=HfApi()
    if api.whoami()['name']!='rayakevin':
        raise ValueError('Compte HF inattendu.')
    preuves=[]
    for dossier, type_repo, identifiant in zip(dossiers,['dataset','model'],
        ['rayakevin/ocr14-medical-bilingue-v3','rayakevin/ocr14-triage-lora-experiences']):
        api.create_repo(identifiant,repo_type=type_repo,private=True,exist_ok=True)
        info=api.repo_info(identifiant,repo_type=type_repo)
        if not info.private:raise ValueError('Le dépôt de remise doit être privé.')
        commit=api.upload_folder(repo_id=identifiant,repo_type=type_repo,folder_path=dossier,
                                 parent_commit=info.sha,commit_message='Livrables OCR14 : partitions v3 et expériences vérifiées')
        confirme=api.repo_info(identifiant,repo_type=type_repo,revision=commit.oid,files_metadata=True)
        preuves.append({'repo':identifiant,'type':type_repo,'revision':confirme.sha,'prive':confirme.private,
                         'fichiers':len(confirme.siblings),'url':str(commit.repo_url)})
    return preuves


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--publier',action='store_true')
    args=p.parse_args()
    dossiers=preparer(args.destination)
    if args.publier:
        preuves=publier(dossiers)
        (RACINE/'reports/publication_livrables_hf.json').write_text(json.dumps(preuves,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(preuves,ensure_ascii=False,indent=2))
    else:print('Paquets préparés :',*dossiers)
