# Parcours du projet

Je présente la démarche en cinq étapes. Les carnets 02 à 05 sont des synthèses
exécutables sur CPU : leurs cellules lisent les bilans versionnés, sans charger
les poids, appeler une API ni relancer les expériences.

| Ordre | Carnet | Question principale |
|---|---|---|
| 01 | [Préparation des données](01_preparation_donnees.ipynb) | Comment construire les corpus et les références ? |
| 02 | [Évaluation](02_evaluation.ipynb) | Comment produire et interpréter les scores ? |
| 03 | [SFT](03_sft.ipynb) | Pourquoi retenir le mélange QA/triage ? |
| 04 | [DPO](04_dpo.ipynb) | Pourquoi ne pas retenir l'alignement testé ? |
| 05 | [Déploiement](05_deploiement.ipynb) | Comment le prototype fonctionne-t-il et quelles preuves sont disponibles ? |
| 06 · Outil | [Lancer la démonstration Colab](06_lancer_demo_colab.ipynb) | Démarrer un endpoint vLLM temporaire, avec GPU |

Je conserve l'ingestion détaillée. Les autres carnets exposent les méthodes,
les scripts utilisés, les résultats déterminants, les décisions et leurs limites.
Les essais intermédiaires sont résumés ; ils ne disparaissent pas des
archives historiques (archives locales non publiées).

Les chiffres viennent de `serving/space/support/resultats.json` et de
`reports/bilan_final_20261105.json`. Le premier conserve les chemins et empreintes
des rapports sources. Les générations, poids et suivis MLflow restent dans les
archives locales et les dépôts HF à accès contrôlé ; un clonage Git ne les fournit pas.
Les sorties des cellules de synthèse correspondent à une relecture CPU des bilans,
pas à une nouvelle validation des modèles.

**Point de méthode :** le test indépendant a évalué le socle et le SFT successif.
Le mixte et le DPO ciblé ont été sélectionnés/comparés sur validation, sans nouveau
test indépendant. Le parcours est organisé par sujet, sans réécrire cette chronologie.

Le [lanceur vLLM Colab](06_lancer_demo_colab.ipynb) reste un
outil opérationnel séparé. Son activation nécessite un GPU et le paquet de
lancement fourni ; il n'est pas nécessaire à la lecture.
