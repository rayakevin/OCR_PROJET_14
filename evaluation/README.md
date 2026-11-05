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
- Je garde le split de test fermé jusqu’à la comparaison d’une étape terminée.
- Je génère sans échantillonnage (`do_sample=False`) afin de rendre les comparaisons reproductibles.
- Je conserve chaque réponse brute en JSONL avant de calculer les métriques.
- Je compare aussi chaque checkpoint sur la tâche précédente afin de détecter un oubli catastrophique.
- Une annotation issue d’un consensus de LLM constitue un standard *silver*, pas une validation clinique.

## Phase 1 — QA médicale générale

| Bloc | Mesure principale | Mesures complémentaires |
|---|---|---|
| MediQAl QCM | accuracy de la lettre A–E | taux de réponse analysable |
| MedQuAD ouvert | F1 lexical | revue humaine de justesse et d’invention |

Les métriques lexicales d’une réponse ouverte ne suffisent pas à établir sa qualité médicale. Elles servent à comparer les versions avec une règle stable ; une revue qualitative complète l’analyse.

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
- Un autre doublon relie validation et test. Je dois exclure sa paire du futur
  bilan indépendant, ainsi que signaler les dix entrées de test déjà exposées
  lors du pilote historique. Le test n'est donc pas entièrement vierge.
- Le comptage d'avertissement reconnaît désormais le pluriel « ne remplacent
  pas » / « do not replace », en plus du singulier. Les rapports initiaux restent
  conservés ; le bilan de rappel recalcule la mesure et expose aussi l'ancien taux.

L'audit des identifiants et des prompts est conservé dans
`data/evaluation/colab_sft_eos_v2/audit_partitions_dpo.json`. Il signale également
quatre prompts du train DPO communs à la validation ou au test QA : ils doivent
être exclus avant le DPO. La présence d'un avertissement et l'accord avec les
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

## Mesure finale préparée après sélection

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
si la sauvegarde peut être achevée. L'exécution GPU reste à confirmer.
