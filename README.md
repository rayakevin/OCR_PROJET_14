# OCR Projet 14 — Prototype de triage médical bilingue

Je développe un prototype pédagogique qui transforme une description de cas médical,
en français ou en anglais, en proposition de niveau ESI et de priorité. Le questionnaire
collecte les informations manquantes ; le modèle classe le cas ; le code contrôle le
format et conserve une trace. Toute proposition nécessite une validation humaine.

**Cas fictifs uniquement. Le projet ne dispose pas de validation clinique indépendante.**

- [Démonstration publique sur Hugging Face](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en)
- [Page explicative interactive](output/Page_explicative_projet.html) : données, SFT, DPO et service ; télécharger le HTML pour utiliser les interactions.
- [Rapport technique](reports/rapport_technique_ocr14.md) · [PDF](output/pdf/rapport_technique_ocr14.pdf)
- [Tests et déploiement GitHub Actions](https://github.com/rayakevin/OCR_PROJET_14/actions/workflows/poc.yml)

## Fonctionnement

```text
Cas fictif FR/EN → questionnaire déterministe → prompt partagé avec l’entraînement
                                                    ↓
                                         Qwen3-1.7B-Base + LoRA
                                                    ↓
                              contrôle du format ESI / priorité / orientation
                                      ↓                         ↓
                            proposition à valider        revue humaine
                                      └──────── trace de l’interaction ────────┘
```

Deux modes de service utilisent le même adaptateur :

| Mode | Stack | Disponibilité |
|---|---|---|
| Interface publique | Gradio, Transformers, PEFT, Hugging Face ZeroGPU | Space public, soumis aux quotas GPU |
| API | FastAPI, vLLM, Docker, SQLite | Exécution locale sur GPU ou endpoint HTTPS temporaire sur Colab |

Le Space n’utilise pas vLLM. L’endpoint Colab reçoit une nouvelle URL à chaque
lancement et s’arrête automatiquement ; aucun endpoint vLLM permanent n’est maintenu.
Les contrôles portent sur le contrat technique, pas sur la justesse médicale.

## Données et méthode

Je construis des jeux de QA médicale, de triage bilingue et de préférences à partir
de corpus documentés. Les annotations ESI reposent sur un consensus de modèles :
ce sont des références *silver*. Les traductions FR/EN d’un même cas restent dans
la même partition pour limiter les fuites.

| Jeu SFT | Entraînement | Validation | Test |
|---|---:|---:|---:|
| QA médicale | 2 098 | 262 | 262 |
| Triage bilingue | 1 888 | 238 | 226 |

Les données et les poids ne sont pas suivis par Git. Ils sont versionnés dans les
dépôts Hugging Face privés [dataset](https://huggingface.co/datasets/rayakevin/ocr14-medical-bilingue-v3)
et [adaptateurs](https://huggingface.co/rayakevin/ocr14-triage-lora-experiences), et fournis
dans la remise. Ces liens nécessitent un droit d’accès. Les sources, licences,
transformations et révisions sont documentées dans leurs fiches et dans le rapport.
Un clonage Git seul ne contient donc pas les corpus ni les archives MLflow.

## Modèle retenu et résultats

Je retiens le **SFT mixte époque 3**, entraîné avec LoRA sur l’union QA/triage.
Le SFT successif, QA puis triage, retrouvait davantage d’urgences sur validation
mais avait fortement dégradé la QA. Le mélange offre un compromis entre les tâches.

| Validation | SFT successif v3 | SFT mixte époque 3 |
|---|---:|---:|
| Macro-F1 des priorités | 0,7715 | 0,7874 |
| Urgences retrouvées | 95/100 | 91/100 |
| QCM corrects | 3,82 % | 57,25 % |

Les essais DPO ciblés réduisent le sur-triage mais abaissent le rappel urgent à
87 %, puis 83 %. Je ne les déploie pas. Leurs poids sont conservés comme variantes
expérimentales ; les poids finaux du DPO généraliste ne sont pas disponibles.

Le **test final concerne uniquement le SFT successif v3** : macro-F1 priorité
0,7858 et 88 urgences retrouvées sur 98. La sélection ultérieure du mixte et les
comparaisons DPO reposent sur la validation, pas sur un nouveau test indépendant.
Les paires traduites sont corrélées et les références ne sont pas des labels cliniques.

## Organisation du dépôt

| Répertoire | Contenu |
|---|---|
| `data_ingestion/` | Exploration, annotation, transformation et partitions |
| [training/](training/README.md) | SFT LoRA, préparation des préférences et DPO |
| [evaluation/](evaluation/README.md) | Inférence, métriques et protocoles |
| [notebooks/](notebooks/README.md) | Cinq carnets explicatifs et lanceur Colab 06 |
| [serving/](serving/README.md) | API, interface, audit, vLLM et Space |
| `tests/` | Tests CPU du pipeline et des contrats de service |
| `.github/workflows/` | Vérification et déploiement automatisés |
| `reports/` | Rapport, bilans agrégés et présentation graphique du projet |

## Parcours de lecture

Je présente le projet dans [cinq carnets](notebooks/README.md) :

1. [Préparation des données](notebooks/01_preparation_donnees.ipynb).
2. [Évaluation : données, métriques et protocole](notebooks/02_evaluation.ipynb).
3. [SFT : successif, mixte et choix du modèle](notebooks/03_sft.ipynb).
4. [DPO : essais, résultats et décision](notebooks/04_dpo.ipynb).
5. [Déploiement : architecture, CI/CD et vérifications](notebooks/05_deploiement.ipynb).

Les carnets 02 à 05 relisent les bilans sur CPU, sans entraînement ni inférence.
Les essais intermédiaires sont résumés et leurs journaux complets (archives locales non publiées)
sont conservés à part. Le lanceur Colab reste un outil opérationnel séparé.

## Exécution et vérification

L’environnement de travail est verrouillé dans `uv.lock` :

```bash
uv sync --frozen
```

Les [instructions de service](serving/README.md) décrivent le chargement de
l’adaptateur, les variables d’environnement, Docker et les appels API.
Le notebook [06](notebooks/06_lancer_demo_colab.ipynb) permet un lancement
Colab ; il nécessite le paquet `ocr14_endpoint_soutenance_20261110.zip`, fourni
avec le notebook dans la remise, à copier dans `MyDrive/OCR_PROJET_14/`.

Le pipeline GitHub Actions exécute les tests Pytest, construit l’image Docker,
puis publie une liste contrôlée de fichiers sur le Space. Le déploiement dépend
du succès des vérifications ; le secret `HF_TOKEN` reste dans GitHub Actions.
La suite comprend 52 tests ; le résultat de chaque publication est consultable dans GitHub Actions.

L’API vLLM publique a été vérifiée le 10 novembre sur L4 avec le SFT mixte :
médiane HTTP de 1,23 s sur trois cas fictifs et huit appels simultanés en 1,54 s.
Ces observations ponctuelles ne sont ni un test de charge ni une garantie de service.

## Suivi expérimental et limites

Les paramètres, pertes et évaluations sont enregistrés dans MLflow. Après
récupération des archives locales, la commande suivante ouvre une copie de consultation :

```bash
.venv/bin/python scripts/ouvrir_mlflow.py --experience mixte
```

Le serveur écoute sur localhost. Les archives d’expériences sont hors Git et ne sont
pas fournies par le Space. Les [protocoles d’évaluation](evaluation/README.md)
détaillent la provenance et les limites de récupération des checkpoints.

Une sortie bien formée peut inventer une donnée clinique. Le questionnaire reste
déterministe et ne constitue pas un protocole médical. Le journal SQLite n’est
ni chiffré ni immuable ; la clé partagée n’identifie pas chaque utilisateur.

Le verrou des expériences n’est pas un socle de production : des alertes de
dépendances liées à `datasets` / `fsspec` et `cryptography` restent à traiter.
Les dépendances du Space et de l’image API sont séparées. Un usage durable demande
une revue de sécurité, une gouvernance des données et une validation clinique indépendante.
