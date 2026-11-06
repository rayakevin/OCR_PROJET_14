# Soutenance — parcours de 15 minutes

Je présente depuis l’onglet **Soutenance · 15 minutes (FR)** du
[Space](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en).
Le support reste lisible sans GPU ; je réserve l’inférence à la démonstration.

| Temps | Écran | Message et preuve |
|---|---|---|
| 0:00–1:30 | Mission | Je vise une aide à la qualification bilingue. Le soignant garde la décision. Les labels sont silver, pas une validation clinique. |
| 1:30–4:00 | Données | Je pars des formats observés, justifie QA / triage / préférences, puis montre consensus, traduction et séparation par cas. 4 974 entrées, 3 986 de train. |
| 4:00–6:30 | Entraînements | Je montre le passage successif, l’oubli QA, le mélange, la poursuite puis le DPO. Le DPO apprend une préférence relative, pas un seuil clinique. |
| 6:30–9:00 | Résultats | Je compare QA, macro-F1, rappel urgent et sur-triage. Le mixte récupère la QA ; le DPO ciblé est refusé car 91 → 87 → 83 urgences retrouvées sur 100. |
| 9:00–11:00 | Pipeline | Je fais avancer une requête. Le questionnaire est un script ; le modèle produit une proposition ; le parseur et l’audit encadrent son usage. |
| 11:00–13:00 | Démonstration | Je reviens au premier onglet : cas fictif, question suivante, sortie, détails et trace JSON. Je distingue le moteur ZeroGPU de l’API vLLM. |
| 13:00–14:00 | MLflow | Je montre les paramètres, la loss et les évaluations. Je relie les résultats aux poids et aux partitions ; les logs API sont un autre suivi. |
| 14:00–15:00 | Décision | Je conclus sur la faisabilité technique et les limites. La suite utile est une évaluation indépendante relue par des professionnels. |

## Préparation

Je démarre MLflow local avant l’oral avec
`.venv/bin/python scripts/ouvrir_mlflow.py --experience dpo-cible`.
Le lien du support vise mon poste ; il ne fournit pas un service MLflow public.
Je vérifie l’accès de l’évaluateur aux dépôts privés et garde le PDF et la page HTML
locale comme repli si la connexion ou le quota ZeroGPU empêche une démo.

Je prépare deux cas explicitement fictifs, sans donnée personnelle. Je ne prédis
pas à l’avance leur ESI pour les faire apparaître comme des résultats validés.
Si une sortie est incohérente ou refusée, je montre le mécanisme de revue humaine.

## Éléments à distinguer pendant la discussion

- Le modèle déployé est le **SFT successif v3** ; la référence expérimentale
  équilibrée est le **SFT mixte époque 3**. Le dernier DPO n’est pas promu.
- Les comparaisons mixtes/DPO ciblé utilisent la **validation**, pas le test historique.
- Les 100 entrées urgentes contiennent des paires traduites, donc des observations corrélées.
- La perte de validation mesure un objectif statistique ; le rappel et le sur-triage
  rendent visibles des erreurs métier différentes.
- L’API vLLM a été testée réellement dans Colab ; le Space public appelle Transformers.
  L’exposition vLLM cloud reste à vérifier avant de la présenter comme livrable livré.
- La démonstration technique ne démontre pas de bénéfice clinique en situation réelle.

## Repli si le temps manque

Je garde les écrans Mission, Données, Résultats, Pipeline, Démo et Décision.
Les paramètres détaillés et les courbes restent disponibles pour les dix minutes
de discussion. Je ne remplace pas la démonstration par la lecture du rapport.
