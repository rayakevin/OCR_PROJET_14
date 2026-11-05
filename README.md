# OCR Projet 14 — POC de triage médical bilingue

Je pars de corpus médicaux hétérogènes pour entraîner puis évaluer un modèle
Qwen3-1.7B-Base adapté avec LoRA. Le projet reste un prototype pédagogique.

## Parcours de lecture

1. `data_ingestion/01_preparation_donnees.ipynb` : exploration, annotations et corpus.
2. `evaluation/02_evaluation_initiale.ipynb` : socle, métriques et protocole.
3. `evaluation/03_execution_colab.ipynb` : corrections, SFT, pilote DPO et comparaison.
4. `serving/04_poc_et_deploiement.ipynb` : questionnaire, API, traces et préparation du déploiement.

Le SFT v3 reste le candidat principal : le pilote DPO n'améliore pas le triage.
La performance QA générale demeure très dégradée après la spécialisation. Les
limites et les erreurs observées sont exposées dans les notebooks.

Je distingue le code testé localement, l'inférence réellement exécutée et le
déploiement vérifié. La démonstration est accessible sur
[Hugging Face ZeroGPU](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en) ;
deux parcours fictifs FR/EN ont produit une réponse réelle et une trace exportable.
La vérification vLLM reste distincte et à réaliser. Aucune validation clinique
n'est revendiquée.

Voir [le POC](serving/README.md), [l'évaluation](evaluation/README.md) et
[l'entraînement](training/README.md) pour les commandes et la provenance.

Les expériences Colab récupérées sont dans `data/colab_archive/`, avec une copie
MLflow locale et un inventaire vérifié. La section 20 du notebook 03 décrit
les accès et la récupération encore incomplète des derniers fichiers DPO.
