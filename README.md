# OCR Projet 14 — POC de triage médical bilingue

Je pars de corpus médicaux hétérogènes pour entraîner puis évaluer un modèle
Qwen3-1.7B-Base adapté avec LoRA. Le projet reste un prototype pédagogique.

## Parcours de lecture

1. `data_ingestion/01_preparation_donnees.ipynb` : exploration, annotations et corpus.
2. `evaluation/02_evaluation_initiale.ipynb` : socle, métriques et protocole.
3. `evaluation/03_execution_colab.ipynb` : corrections, SFT, pilote DPO et comparaison.
4. `serving/04_poc_et_deploiement.ipynb` : questionnaire, API, traces et préparation du déploiement.
5. `evaluation/05_finalisation_colab.ipynb` : test final, sauvegarde et libération du GPU.
6. `evaluation/06_verification_vllm_colab.ipynb` : reprise vLLM/API sur port libre, sans recalculer le test.

Le SFT v3 reste le candidat principal : le pilote DPO n'améliore pas le triage.
La performance QA générale demeure très dégradée après la spécialisation. Les
limites et les erreurs observées sont exposées dans les notebooks.

Je distingue le code testé localement, l'inférence réellement exécutée et le
déploiement vérifié. La démonstration est accessible sur
[Hugging Face ZeroGPU](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en) ;
deux parcours fictifs FR/EN ont produit une réponse réelle et une trace exportable.
La voie vLLM/API est vérifiée sur trois cas fictifs dans Colab (A100 40 Go),
avec une revue humaine due à un libellé anglais non canonique. Elle reste interne
à une VM temporaire. Aucune validation clinique n'est revendiquée.

Voir [le POC](serving/README.md), [l'évaluation](evaluation/README.md) et
[l'entraînement](training/README.md) pour les commandes et la provenance.

Les expériences Colab récupérées sont dans `data/colab_archive/`, avec une copie
MLflow locale et un inventaire vérifié. La section 20 du notebook 03 décrit
les accès et la récupération encore incomplète des derniers fichiers DPO.

Le [pipeline GitHub Actions](https://github.com/rayakevin/OCR_PROJET_14/actions/runs/42074793162)
a exécuté avec succès 35 tests, construit l'image Docker et publié le paquet
vérifié sur le Space existant. Il utilise le secret `HF_TOKEN` du dépôt.

Le [rapport technique](reports/rapport_technique_ocr14.md) et sa
[version PDF](output/pdf/rapport_technique_ocr14.pdf) distinguent les résultats
du test final des expériences de validation. Le prototype n'est pas présenté
comme un système cliniquement validé ni comme un déploiement vLLM public permanent.

## Bilan final

Les 976 prédictions du socle et du SFT sont récupérées et les métriques recalculées
localement. Sur les 226 triages, le SFT atteint 0,7858 de macro-F1 priorité et
retrouve 88/98 urgents de référence. Les 10 sous-triages critiques concernent
7 cas distincts. La QA reste dégradée : 0/131 QCM corrects.

Les trois requêtes vLLM/API ont une médiane HTTP de 514 ms ; cette petite mesure
n'est pas un test de charge. Les poids finaux DPO restent absents de la sauvegarde
accessible. Je conserve cet écart et ne substitue pas un autre checkpoint.
La synthèse et les empreintes sont dans `reports/bilan_final_20261105.json`.
