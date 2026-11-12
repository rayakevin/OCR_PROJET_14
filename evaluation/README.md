# Protocole d’évaluation

Je mesure les quatre états du modèle avec les mêmes données, le même prompt et les mêmes paramètres de génération :

```text
Qwen3-1.7B-Base
        ↓
LoRA après la phase 1 — QA médicale
        ↓
LoRA après la phase 2 — triage
        ↓
Modèle après DPO
```

## Règles communes

- J’utilise le split de validation pour choisir les hyperparamètres et le checkpoint.
- Le test final a été ouvert le 5 novembre pour le socle et le SFT successif ; les variantes suivantes sont comparées sur validation.
- Je génère sans échantillonnage (`do_sample=False`) afin de rendre les comparaisons reproductibles.
- Je conserve chaque réponse brute en JSONL avant de calculer les métriques.
- Je compare aussi chaque checkpoint sur la tâche précédente afin de détecter un oubli catastrophique.
- Une annotation issue d’un consensus de LLM constitue un standard *silver*, pas une validation clinique.

## Phase 1 — QA médicale générale

| Bloc | Mesure principale | Mesures complémentaires |
|---|---|---|
| MediQAl QCM | accuracy de la lettre A–E | taux de réponse analysable |
| MedQuAD ouvert | F1 lexical | revue clinique indépendante à réaliser |

Les métriques lexicales d’une réponse ouverte ne suffisent pas à établir sa qualité médicale. Elles servent à comparer les versions avec une règle stable ; une revue qualitative ne remplace pas une validation clinique indépendante.

## Phase 2 — triage

Le cahier des charges formule la décision métier avec trois priorités. La **macro-F1 sur ces trois priorités** est donc la métrique principale : elle donne le même poids à chaque classe malgré leur fréquence différente.

| Famille | Métriques |
|---|---|
| Priorité métier | macro-F1, accuracy, rappel par classe, IC de Wilson à 95 %, matrice de confusion |
| ESI détaillé | accuracy exacte, macro-F1 à cinq classes, erreur absolue moyenne |
| Sécurité | sous-triage, sous-triage critique, sur-triage, rappel de l’urgence maximale |
| Comportement | sortie ESI analysable, avertissement médical, langue et format attendus |
| Bilingue | cohérence de la décision entre les deux langues d’un même cas |

Le sous-triage critique désigne ici un cas ESI 1 ou 2 prédit au niveau 3, 4 ou 5. Cette définition est une convention d’évaluation du POC et ne vaut pas validation clinique.

## Commandes

Pilote d’inférence :

```bash
python evaluation/inference.py \
  --phase triage --split test --limit 10 \
  --output data/evaluation/baseline_qwen3_1_7b_base/triage_test_pilote.jsonl
```

Calcul des métriques à partir des réponses conservées :

```bash
python evaluation/evaluer.py \
  data/evaluation/baseline_qwen3_1_7b_base/triage_test.jsonl \
  --phase triage --split test
```

Les mêmes commandes seront réutilisées en changeant seulement le checkpoint et le dossier de sortie.

## Comparaison après la phase 1

Je compare d'abord le modèle de base et l'adaptateur sur les mêmes jeux de
validation : 262 exemples QA et 238 exemples de triage. Cette mesure est
exploratoire : la validation QA a aussi servi à choisir le meilleur checkpoint.
Je conserve le test complet pour la comparaison finale.

Le paramètre `--adapter` charge les poids LoRA sur le socle. Je conserve le
tokenizer du socle pour ne pas changer le prompt entre les deux mesures.

```bash
uv run python evaluation/inference.py \
  --phase qa --split validation --batch-size 4 --max-new-tokens 256 \
  --adapter models/phase1_qa_lora_kernel38_20261105_093746/adapter \
  --output data/evaluation/phase1_qa_lora_kernel38_20261105_093746/qa_validation.jsonl

uv run python evaluation/evaluer.py \
  data/evaluation/phase1_qa_lora_kernel38_20261105_093746/qa_validation.jsonl \
  --phase qa --split validation
```

Pour le triage, je remplace `qa` par `triage` et limite la sortie à 192 tokens,
comme pour la référence initiale. Les commandes réécrivent les prédictions :
je choisis un nouveau dossier pour conserver un essai antérieur.

Après une interruption, j'ajoute `--resume` pour conserver les lots complets.
Le script vérifie leur ordre et leur configuration avant de reprendre ; un
fichier tronqué ou un lot incomplet doit être examiné avant toute relance.
Je conserve la même taille de lot et vérifie les empreintes du protocole.

Le notebook présente le tableau de comparaison et ses limites. Le dossier de
résultats conserve aussi `protocole.json` : empreintes des données, de l'adaptateur,
des scripts et des prédictions de référence, paramètres et identifiants MLflow.

## Contrôles complémentaires après le SFT de triage

Je compare le triage seul et un rappel de QA dans la section 12 de
`03_execution_colab.ipynb`. La génération reste identique ; les mêmes prédictions
servent aux métriques historiques et à l'analyse sans doublon.

- Une paire bilingue de validation partage son prompt anglais avec le train sous
  un autre identifiant. Je présente aussi les scores sur 236 entrées, en excluant
  cette paire, et je retire la paire correspondante du train du nouvel essai.
- Un autre doublon relie validation et test. Cette paire est ensuite exclue du
  bilan v3. Les dix entrées déjà exposées au pilote historique sont également
  exclues : le test historique initial n’était pas entièrement vierge.
- Le comptage d'avertissement reconnaît désormais le pluriel « ne remplacent
  pas » / « do not replace », en plus du singulier. Les rapports initiaux restent
  conservés ; le bilan de rappel recalcule la mesure et expose aussi l'ancien taux.

L'audit des identifiants et des prompts est conservé dans
`data/evaluation/colab_sft_eos_v2/audit_partitions_dpo.json`. Il signale également
quatre prompts du train DPO communs à la validation ou au test QA : ils sont ensuite exclus dans les partitions v3 avant le DPO. La présence d'un avertissement et l'accord avec les
annotations ne constituent pas une validation clinique.

## Version v3 : partitions figées et comparaison SFT / DPO

L'audit précédent est maintenant appliqué dans `data/processed/partitions_v3`.
Je retire les doublons par paire bilingue et les cinq paires de test déjà
exposées au pilote. Le test triage restant contient 226 entrées ; je le réserve
à la comparaison finale, après le choix du modèle sur validation.

Les scripts acceptent `--data-dir data/processed/partitions_v3`. L'inférence
enregistre l'empreinte de la partition v3 et refuse une reprise incompatible.
Les sections 15 à 18 du notebook 03 reconstruisent les données, reprennent le
SFT de triage et évaluent son prolongement par un pilote DPO sur les mêmes
262 exemples QA et 238 exemples de triage. Les résultats restent séparés des
expériences historiques, dans `experiences/sft_dpo_v3_20261105` sur Drive.

Je rapporte le nombre de sous-triages critiques et deux dénominateurs distincts :
les sorties ESI analysables pour la métrique historique, et les cas urgents de
référence pour interpréter les erreurs manquées. Une sortie non analysable
reste une erreur de classification, même si elle n'entre pas dans le premier
dénominateur. Les scores par langue et l'accord des traductions complètent
la moyenne globale.

## Récupération locale de Colab

Je conserve les fichiers téléchargés dans `data/colab_archive/original/`, avec
un inventaire Drive et les empreintes de transfert. Le notebook local conserve
ses analyses les plus récentes ; la copie distante reste disponible dans l’archive.

```bash
.venv/bin/python evaluation/restaurer_colab.py
.venv/bin/mlflow ui --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///data/colab_archive/local/mlflow_colab.db
```

La copie MLflow adapte les URI des artefacts au disque local sans modifier la
base originale ni le suivi des entraînements locaux précédents. Les checkpoints
sont dans `original/experiences/`. Le rapport `verification.json` constate
l’intégrité de la copie ; les limites d’exhaustivité figurent dans le notebook 03.

## Mesure finale après sélection

Le candidat SFT v3 est figé avant la lecture du test. Le protocole versionné
`protocole_test_final.json` conserve les empreintes des 262 QA et des 226 triages.
`test_final.py` refuse d'autres données ou poids, appelle la même inférence et
conserve prédictions, métriques et artefacts MLflow. Le DPO n'est ajouté que si
ses poids originaux sont retrouvés et correspondent à leur empreinte.

Le notebook `05_finalisation_colab.ipynb`, également déposé dans Drive, regroupe
ces mesures avec la vérification vLLM dans un environnement Python séparé. Il ne
relance aucun entraînement. Les sorties et les erreurs sont sauvegardées dans
`evaluations/finalisation_20261105` sur Drive, puis réunies dans une archive.
La synchronisation Drive précède la libération de la VM, même après une erreur
si la sauvegarde peut être achevée. L'exécution est terminée pour le socle et le SFT : 976 prédictions, quatre runs
MLflow terminés et métriques recalculées identiques en local. Le DPO reste absent
du test final faute de poids récupérables. La vérification API est reprise
séparément dans le notebook 06, après correction du conflit de port Colab.

Les preuves sont dans `data/colab_archive/finalisation_20261105/`, et la seconde
copie MLflow, avec ses liens locaux, dans `data/colab_archive/local/mlflow_finalisation.db`.
Le bilan publiable sans corpus est `reports/bilan_final_20261105.json`.

```bash
.venv/bin/mlflow ui --host 127.0.0.1 --port 5002 \
  --backend-store-uri sqlite:///data/colab_archive/local/mlflow_finalisation.db
```

## Variante : SFT mixte depuis le socle

Dans [07_sft_mixte_colab.ipynb](07_sft_mixte_colab.ipynb), je prépare un nouveau
LoRA sur l'union QA/triage des partitions v3 : 3 986 exemples d'entraînement
et 500 de validation. Je conserve trois checkpoints et mesure les deux tâches
séparément, avec le même protocole de génération que le SFT v3. Le test final
n'est pas inclus dans le paquet Colab. Cette expérience a été exécutée le 6 novembre. L’époque 3 est retenue et
déployée depuis le 10 novembre ; sa sélection repose sur la validation.

## Lecture actuelle et DPO ciblé

Je lis 01 → 02 → 03 pour la construction de l'expérience, 04 → 05 → 06 pour
la démonstration et les preuves historiques, puis 07 → 08 pour les variantes.
Les encadrés « Repère de lecture » distinguent les résultats historiques des décisions actuelles.

Le test final a été consulté le 5 novembre : il ne constitue plus un jeu de sélection
pour les variantes suivantes. Le notebook 08 n'inclut aucun fichier de test dans son paquet.
Il réévalue le SFT mixte époque 3 et les pas DPO 16/32 sur les mêmes 500 entrées de validation.
Les seuils de comparaison sont définis dans `comparer_dpo_triage.py` avant le calcul.
Le rappel urgent doit augmenter sans dégradation excessive du QA, de la macro-F1,
du sur-triage ou du format. Une hausse marginale n'est pas une preuve clinique.

Les courbes archivées se consultent localement, après récupération des archives :
` .venv/bin/python scripts/ouvrir_mlflow.py --experience mixte `.


## Décision après le DPO ciblé

Le run du 6 novembre est terminé : 355/85 paires, 32 pas. Sur validation,
le rappel urgent passe de 91 % pour le SFT mixte réévalué à 87 % puis 83 %,
et la macro-F1 de 0,7875 à 0,7457 puis 0,7373. Je ne retiens aucun de ces DPO.
Les 1 500 prédictions, six rapports, poids et états d’entraînement sont récupérés ;
95 empreintes correspondent. Le notebook 08 conserve le bilan et les limites.
