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
7. `evaluation/07_sft_mixte_colab.ipynb` : SFT commun QA/triage, époques 1 à 10 et décision.
8. `evaluation/08_dpo_triage_cible_colab.ipynb` : DPO ciblé sur les erreurs du train, résultats et décision de non-sélection.
9. `evaluation/09_endpoint_soutenance_colab.ipynb` : endpoint vLLM HTTPS temporaire, exécuté et vérifié le 10 novembre sur L4 ; à relancer avant l’oral.

Le **modèle retenu et déployé est le SFT mixte époque 3** (Space et configuration
de l'API depuis le 10 novembre). Le **SFT successif v3**, déployé auparavant, reste
le seul modèle mesuré sur le test final. Le mélange restaure largement la QA et améliore légèrement
la macro-F1 de triage, avec un rappel urgent inférieur au SFT successif sur validation (91 contre 95 sur 100).
La poursuite jusqu'à dix époques n'apporte pas de meilleur compromis global.
Le DPO généraliste n'a pas amélioré le triage ; le DPO ciblé du notebook 08 réduit le sur-triage mais dégrade le rappel urgent
(91 → 87 → 83 %) : il n’est pas retenu.

Les notebooks suivent la chronologie des décisions. Je lis les résultats conservés
sans réexécuter les anciens calculs. Les notebooks 07 et 08 ont un interrupteur de
lancement désactivé par défaut ; chaque nouvelle expérience utilise un autre dossier.
La numérotation est un ordre de lecture, pas une commande « tout réentraîner ».

Je distingue le code testé localement, l'inférence réellement exécutée et le
déploiement vérifié. La démonstration est accessible sur
[Hugging Face ZeroGPU](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en) ;
après la bascule sur le SFT mixte, un parcours fictif y produit une proposition réelle
(révision `f32b2a65` du Space), et deux parcours FR/EN sont vérifiés en local.
La voie vLLM/API a été vérifiée le 5 novembre sur trois cas fictifs dans Colab (A100 40 Go)
avec le SFT successif, avec une revue humaine due à un libellé anglais non canonique.
Le 10 novembre, le notebook 09 l'expose en HTTPS avec le SFT mixte sur une L4 Colab :
médiane HTTP 1,23 s sur trois cas fictifs, huit requêtes simultanées en 1,54 s
(5,2 requêtes/s, 329 tokens/s), refus 401 sans clé vérifié depuis l'extérieur. Cet endpoint
est temporaire : son URL change à chaque lancement. Aucune validation clinique n'est revendiquée.

Voir [le POC](serving/README.md), [l'évaluation](evaluation/README.md) et
[l'entraînement](training/README.md) pour les commandes et la provenance.

Les expériences Colab récupérées sont dans `data/colab_archive/`, avec une copie
MLflow locale et un inventaire vérifié. La section 20 du notebook 03 décrit
les accès et la récupération encore incomplète des derniers fichiers DPO.

Le [pipeline GitHub Actions](https://github.com/rayakevin/OCR_PROJET_14/actions/workflows/poc.yml)
exécute la suite Pytest (52 tests), construit l'image Docker puis publie le paquet
vérifié sur le Space existant ; le déploiement ne démarre que si les tests réussissent. Il utilise le secret `HF_TOKEN` du dépôt.

Le [rapport technique](reports/rapport_technique_ocr14.md) et sa
[version PDF](output/pdf/rapport_technique_ocr14.pdf) distinguent les résultats
du test final des expériences de validation. Le prototype n'est pas présenté
comme un système cliniquement validé ni comme un déploiement vLLM public permanent.

## Bilan historique du SFT successif (5 novembre)

Les 976 prédictions du socle et du SFT sont récupérées et les métriques recalculées
localement. Sur les 226 triages, le SFT atteint 0,7858 de macro-F1 priorité et
retrouve 88/98 urgents de référence. Les 10 sous-triages critiques concernent
7 cas distincts. La QA reste dégradée : 0/131 QCM corrects.

Les trois requêtes vLLM/API ont une médiane HTTP de 514 ms ; cette petite mesure
n'est pas un test de charge. Les poids finaux DPO restent absents de la sauvegarde
accessible. Je conserve cet écart et ne substitue pas un autre checkpoint.
La synthèse et les empreintes sont dans `reports/bilan_final_20261105.json`.

## Ouvrir MLflow pour la démonstration

Depuis la racine du dépôt, après installation de l'environnement (`uv sync --frozen`) :

```bash
.venv/bin/python scripts/ouvrir_mlflow.py
```

Le serveur démarre sur `http://127.0.0.1:5000` et ouvre automatiquement le navigateur.
Je peux afficher les paramètres, la loss et les runs d'évaluation du SFT mixte.
`Ctrl+C` arrête le serveur ; aucun GPU n'est utilisé, aucun accès public n'est créé.

```bash
.venv/bin/python scripts/ouvrir_mlflow.py --experience historique
.venv/bin/python scripts/ouvrir_mlflow.py --experience poursuite
.venv/bin/python scripts/ouvrir_mlflow.py --experience final
.venv/bin/python scripts/ouvrir_mlflow.py --experience dpo-cible
```

Je choisis une base à la fois ; `--port 5001` permet une deuxième fenêtre.
Les archives originales restent intactes : l'outil prépare une copie de consultation.
Les paramètres et métriques sont récupérés ; les artefacts et poids des derniers
essais ne sont pas tous présents localement. Une courbe visible ne prouve pas une
sauvegarde complète des checkpoints. Les archives ignorées par Git doivent être
récupérées avant cette démonstration sur une autre machine.

## Dernière expérience terminée

Le notebook 08 mine uniquement `phase2_triage_train.jsonl` avec le checkpoint 750
mixte. Il sépare les erreurs par cas bilingue en train DPO et contrôle interne,
puis teste 32 mises à jour contre la référence SFT figée. Le bilan compare rappel
urgent, sur-triage, macro-F1, QA et format sur les validations v3. Les 1 500 prédictions sont récupérées et leurs métriques recalculées ;
les deux checkpoints échouent aux critères prévus. Le Space sert le SFT mixte époque 3, source de ce DPO. Le test historique a déjà été vu ;
une confirmation indépendante des nouvelles variantes demanderait de nouveaux cas.

## Conformité aux livrables et préparation de la soutenance

Je confronte les productions à la consigne « Livrables et soutenance » fournie le
6 novembre 2026. Une expérience réussie ne suffit pas à constituer un livrable déposé.

| Livrable demandé | État vérifié | Action restante |
|---|---|---|
| Dataset bilingue JSONL / Hugging Face Datasets | [Dépôt HF privé](https://huggingface.co/datasets/rayakevin/ocr14-medical-bilingue-v3), quatre configurations chargées depuis la révision publiée | Vérifier l’accès de l’évaluateur ; restrictions de licence documentées |
| Modèle spécialisé SFT/LoRA et activité DPO | [Adaptateurs HF privés](https://huggingface.co/rayakevin/ocr14-triage-lora-experiences) : mixte (déployé), successif, DPO ciblé 16/32 ; dernier DPO non retenu | Republier la carte du dépôt après la bascule du 10 novembre |
| Rapport technique PDF, 20 pages maximum | 19 pages : choix techniques, risques, coûts, endpoint public, conditions go / no-go | Relecture et dépôt du PDF |
| Endpoint cloud utilisant vLLM | Endpoint HTTPS public vérifié le 10 novembre (Colab L4, SFT mixte, latence et débit mesurés, journaux conservés) | Relancer le notebook 09 une vingtaine de minutes avant l'oral ; pas d'hébergement permanent |
| CI/CD GitHub Actions | Pytest, Docker et publication du Space | Vérifier l'exécution sur la révision livrée après le commit |

Le texte n'impose pas un GPU allumé en permanence : je peux prévoir un service
activable pour la soutenance, avec une procédure de démarrage et de contrôle.
En revanche, expliquer l'écart ne rend pas le Space Transformers équivalent au
livrable vLLM demandé. La journalisation corrèle désormais les appels entrants, les refus 401/422 et les appels vLLM, sans exposer les clés ni les corps dans les logs techniques.

Je prépare ensuite le ZIP `Titre_du_projet_nom_prenom`, avec les noms de fichiers
`Nom_Prenom_n°_nom_du_livrable_mmaaaa`. Le mois correspond au démarrage du projet,
pas à la date d'export. Le conditionnement utilise Raya_Kevin et 102026, sur la base du premier commit du 16 octobre 2026.
Le ZIP doit permettre de retrouver données, poids, rapport, endpoint et code CI/CD,
sans secrets ni documents internes étrangers au projet.

Pour l'oral de 15 minutes, je réserve 2 minutes au besoin et aux limites, 3 aux
données et au consensus, 4 aux entraînements et métriques, 4 à la démonstration,
puis 2 aux perspectives et à la décision. Les 10 minutes de discussion portent
sur validation, fine-tuning, passage à l'échelle et déploiement ; le débrief dure
5 minutes. Je présente une valeur clinique potentielle, soutenue par les résultats
sur références silver, sans revendiquer une efficacité clinique démontrée.

## Support de soutenance intégré

L’onglet **Soutenance · 15 minutes (FR)** du Space réunit le parcours, les schémas
interactifs, les comparaisons et un export des courbes MLflow. Les animations ne
consomment pas de GPU. Le [plan oral](reports/plan_soutenance.md) donne le minutage.
Les graphiques publics ne contiennent aucun cas individuel. La page
[Soutenance triage OCR14](reports/Soutenance%20triage%20OCR14.html) est le support principal de l'oral.

Les révisions de remise et leurs URL figurent dans
[publication_livrables_hf.json](reports/publication_livrables_hf.json).
Le chargement distant des quatre configurations a été vérifié : QA, triage,
DPO général et DPO ciblé. Les colonnes optionnelles ont été uniformisées avec null,
sans modification des textes, des identifiants ou des partitions.

Les dépôts HF sont privés ; le lien seul ne donne pas accès à l’évaluateur.
Je ne rends pas les données publiques pour contourner cette question d’accès.


## Périmètre des dépendances et de la sécurité

La revue Dependabot a signalé des versions anciennes dans l’environnement de
travail. Flask, Gunicorn et Waitress ont été actualisés dans le verrou de dépendances.
Les alertes liées à `datasets` / `fsspec` et `cryptography` restent à traiter avec
leurs contraintes amont (Datasets 4.7 et Presidio). Je conserve les versions des
expériences pour la reproductibilité ; ce verrou n’est pas présenté comme un
socle prêt pour la production. MLflow écoute seulement sur localhost. Le paquet
du Space et l’image FastAPI ont leurs dépendances séparées ; cette distinction
ne dispense pas d’un audit de sécurité avant un hébergement durable.
