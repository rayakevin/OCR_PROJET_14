# Évaluation des modèles

Le [notebook 02](../notebooks/02_evaluation.ipynb) présente les métriques et la portée des résultats. Les comparaisons SFT et DPO figurent dans les [carnets 03](../notebooks/03_sft.ipynb) et [04](../notebooks/04_dpo.ipynb).

## Protocole et résultats

Les partitions v3 comprennent 262 QA et 238 triages de validation, puis 262 QA et 226 triages de test. Les traductions d'un même cas restent ensemble. Les références ESI sont issues d'un consensus de LLM, sans validation clinique indépendante.

La validation sert à comparer les variantes. Le test final a évalué le socle et le SFT successif : macro-F1 priorité 0,7858 et 88 urgences retrouvées sur 98 pour ce dernier. Le mixte époque 3, retenu ensuite, a été choisi sur validation : macro-F1 0,7874, rappel urgent 91/100 et QCM 57,25 %. Ces résultats ne constituent pas un nouveau test indépendant.

Les DPO ciblés retrouvent 87 puis 83 urgences sur 100 ; ils ne sont pas retenus. Les résultats agrégés et leur provenance sont dans `../serving/space/support/resultats.json`, et le bilan du test dans `../reports/bilan_final_20261105.json`.

## Métriques

| Dimension | Mesures | Interprétation |
|---|---|---|
| Priorité | Macro-F1, accuracy, rappel par classe, matrice de confusion | Trois classes : maximale, modérée, différée |
| ESI | Exactitude, macro-F1, erreur absolue, taux analysable | Le respect du format ne valide pas le classement |
| Sens des erreurs | Sous-triage, sous-triage critique, sur-triage | Un ESI 1–2 prédit 3–5 est un sous-triage critique |
| QA | Exactitude QCM et F1 lexical des réponses ouvertes | Le F1 lexical ne mesure pas la justesse médicale |
| Bilingue | Accord FR/EN sur le même cas | Les deux entrées sont corrélées |

Le rappel urgent utilise tous les urgents de référence comme dénominateur. Les taux historiques de sous-triage et de sur-triage utilisent les sorties analysables : ils ne doivent pas être confondus avec ce rappel. Une sortie non analysable reste une erreur de classification.

## Générer puis calculer les métriques

L'inférence charge le socle et, si demandé, un adaptateur. Seuls les messages système et utilisateur sont fournis au modèle. Les réponses sont enregistrées en JSONL avant le calcul des métriques. La génération sans échantillonnage limite l'aléa, sans garantir une identité entre matériels et moteurs.

Exemple de nouvelle évaluation sur validation, depuis la racine du dépôt :

```bash
uv run python evaluation/inference.py \
  --phase triage --split validation --data-dir data/processed/partitions_v3 \
  --adapter CHEMIN_ADAPTATEUR --batch-size 4 --max-new-tokens 192 \
  --output data/evaluation/nouvelle_evaluation/triage_validation.jsonl

uv run python evaluation/evaluer.py \
  data/evaluation/nouvelle_evaluation/triage_validation.jsonl \
  --phase triage --split validation --data-dir data/processed/partitions_v3
```

Les corpus et poids sont requis mais absents d'un simple clonage Git. Choisir un nouveau dossier de sortie conserve les essais précédents. `--resume` reprend les lots complets après contrôle de l'ordre et de la configuration ; un fichier tronqué demande une vérification préalable.

Le script `test_final.py` contrôle les empreintes du protocole `protocole_test_final.json`. Le jeu final déjà consulté ne doit pas servir à sélectionner les nouvelles variantes.

## Traçabilité et archives

Les prédictions, paramètres, empreintes et métriques MLflow sont conservés localement ou dans les archives Colab. `restaurer_colab.py` vérifie les transferts et adapte les chemins dans une copie, sans réécrire les originaux.

```bash
.venv/bin/python scripts/ouvrir_mlflow.py --experience mixte
```

Les archives ne sont pas fournies par Git. Les résultats du DPO généraliste sont disponibles, mais ses poids finaux sont absents de la sauvegarde récupérée ; les poids et résultats du DPO ciblé sont conservés. Les carnets actuels relisent les bilans sur CPU, sans relancer les calculs GPU.
