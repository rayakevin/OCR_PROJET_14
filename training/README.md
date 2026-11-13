# Entraînement SFT et DPO

Les [carnets SFT](../notebooks/03_sft.ipynb) et [DPO](../notebooks/04_dpo.ipynb) présentent les hypothèses, résultats et décisions. Le [rapport](../reports/rapport_technique_ocr14.md) en donne l'analyse technique.

## Modèle retenu

Le SFT mixte époque 3 est le candidat de démonstration. Il apprend conjointement sur 2 098 QA et 1 888 triages, depuis Qwen3-1.7B-Base avec un nouvel adaptateur LoRA. Il conserve mieux la QA que le SFT successif, mais retrouve moins d'urgences : 91 contre 95 sur 100 en validation. Le test indépendant concerne uniquement le successif.

| Paramètre mixte | Valeur |
|---|---|
| Socle figé | Qwen/Qwen3-1.7B-Base |
| Révision | ea980cb0a6c2ae4b936e82123acc929f1cec04c1 |
| LoRA | Rang 16, alpha 32 ; 17 432 576 paramètres |
| Entraînement | Trois époques, batch effectif 16, BF16, seed 3407 |
| Taux maximal / warmup | 0,0001 / 20 pas |
| Contexte maximal | 5 120 tokens |
| Checkpoints | 250, 500, 750 pas ; époque 3 retenue |

Le manifeste `../serving/modele.json` identifie l'adaptateur servi. L'adaptateur seul ne contient pas le socle.

## Scripts et format

`entrainer_lora.py` supervise uniquement les tokens de la réponse assistant et conserve les paramètres et métriques dans MLflow. `format_sft.py` partage le format du prompt avec l'inférence et ajoute l'EOS natif. Les exemples trop longs sont refusés plutôt que tronqués silencieusement.

Exemple de lancement QA depuis la racine, avec les données disponibles localement :

```bash
uv run python training/entrainer_lora.py \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --output models/phase1_qa_eos_v2 --max-length 5120
```

Pour poursuivre un adaptateur sur le triage, le script accepte `--adapter`, `--phase phase2_triage` et les chemins `--train` / `--validation`. Les options exactes sont accessibles avec `--help`. Les jeux d'entraînement et les poids restent hors Git.

## Enseignements des variantes

Le SFT successif poursuit le même LoRA : QA puis triage. Il spécialise le classement, mais la QA se dégrade fortement. Un rappel partiel de QA a conduit à tester le mélange complet. La poursuite du mixte au-delà de l'époque 3 utilise un nouvel optimiseur et un taux réduit ; aucun checkpoint ne domine tous les critères. Ces essais sont résumés dans le carnet 03, leurs journaux détaillés restant archivés localement.

## DPO

`preparer_dpo.py` prépare les préférences générales. Le pilote généraliste (256/64 paires, 32 pas) améliore la loss sans corriger les cinq sous-triages critiques du successif. Ses poids finaux ne sont pas disponibles dans la sauvegarde récupérée.

`preparer_dpo_triage.py` construit ensuite des paires à partir des erreurs du mixte sur le train triage uniquement. Le contrôle interne DPO garde FR/EN ensemble et reste issu du train SFT ; il ne mesure pas une généralisation indépendante. La validation métier utilise les 500 entrées v3 séparées.

`entrainer_dpo.py` conserve une référence SFT figée. Le pilote ciblé utilise 355 paires d'entraînement, 85 de contrôle, 32 pas, un taux de 2e-6, beta 0,1 et un batch effectif de 16. Les checkpoints 16/32 réduisent le sur-triage mais abaissent le rappel urgent à 87/83 %. Aucun n'est déployé.

La sélection par loss du dossier `adapter/` ne remplace pas la comparaison métier des checkpoints. Les essais CPU vérifient les contrats et la référence figée ; ils ne valident pas les performances cliniques.
