# Soutenance — parcours de 15 minutes

Je présente depuis la page `reports/Soutenance triage OCR14.html`, puis je passe au
[Space](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en) pour la démonstration.
L’onglet **Soutenance · 15 minutes (FR)** du Space reste un repli.
Le support reste lisible sans GPU ; je réserve l’inférence à la démonstration.

| Temps | Écran | Message et preuve |
|---|---|---|
| 0:00–1:30 | Mission | Je vise une aide à la qualification bilingue. Le soignant garde la décision. Les labels sont silver, pas une validation clinique. |
| 1:30–4:00 | Données | Je pars des formats observés, justifie QA / triage / préférences, puis montre consensus, traduction et séparation par cas. 4 974 entrées, 3 986 de train. |
| 4:00–6:30 | Entraînements | Je compare les deux SFT : le successif oublie la QA, le mixte garde les deux tâches et je le retiens. Puis la poursuite et les deux DPO. Le DPO apprend une préférence relative, pas un seuil clinique. |
| 6:30–9:00 | Résultats | Je compare QA, macro-F1, rappel urgent et sur-triage. Le mixte récupère la QA ; le DPO ciblé est refusé car 91 → 87 → 83 urgences retrouvées sur 100. |
| 9:00–11:00 | Pipeline | Je fais avancer une requête. Le questionnaire est un script ; le modèle produit une proposition ; le parseur et l’audit encadrent son usage. |
| 11:00–13:00 | Démonstration | Space : cas fictif, question suivante, sortie, détails et trace JSON. Je montre ensuite les preuves de l’endpoint vLLM du 10 novembre (latence, débit, trace). L’appel vLLM en direct se fait seulement si l’évaluateur le demande. |
| 13:00–14:00 | MLflow | Je montre les paramètres, la loss et les évaluations. Je relie les résultats aux poids et aux partitions ; les logs API sont un autre suivi. |
| 14:00–15:00 | Décision | Je conclus sur la faisabilité technique et les limites. La suite utile est une évaluation indépendante relue par des professionnels. |

## Préparation

À 8 h 30 le jour de l’oral, dans cet ordre :

1. Vérifier que le Space répond avec un cas fictif (le premier appel réveille ZeroGPU).
2. Lancer MLflow local (commande ci-dessous).
3. Ouvrir la page de soutenance et le rapport PDF.
4. En bonus : lancer le notebook 09 sur une L4 et noter la clé saisie et l’URL affichée.
   Le service reste ouvert 60 minutes après sa vérification ; si l’oral se termine plus
   tard, je le lance plus près de l’heure de passage.

Je démarre MLflow local avant l’oral avec
`.venv/bin/python scripts/ouvrir_mlflow.py --experience dpo-cible`.
Le lien du support vise mon poste ; il ne fournit pas un service MLflow public.
Je vérifie l’accès de l’évaluateur aux dépôts privés et garde le PDF et la page HTML
locale comme repli si la connexion ou le quota ZeroGPU empêche une démo.

Je prépare deux cas explicitement fictifs, sans donnée personnelle. Je ne prédis
pas à l’avance leur ESI pour les faire apparaître comme des résultats validés.
Si une sortie est incohérente ou refusée, je montre le mécanisme de revue humaine.

## Éléments à distinguer pendant la discussion

- Le modèle déployé est le **SFT mixte époque 3** depuis le 10 novembre. Le
  **SFT successif v3** est le seul mesuré sur le test final. Le dernier DPO n’est pas promu.
- Les comparaisons mixtes/DPO ciblé utilisent la **validation**, pas le test historique.
- Les 100 entrées urgentes contiennent des paires traduites, donc des observations corrélées.
- La perte de validation mesure un objectif statistique ; le rappel et le sur-triage
  rendent visibles des erreurs métier différentes.
- L’endpoint vLLM public a été vérifié le 10 novembre sur une L4 Colab avec le SFT mixte :
  médiane 1,23 s, huit requêtes simultanées en 1,54 s. Le Space public appelle Transformers.
  Je lance le notebook 09 une vingtaine de minutes avant l’oral (installation et chargement
  compris), après avoir vérifié que le paquet du 10 novembre est bien dans Drive. Une L4 suffit.
  L’URL change à chaque lancement et le service s’arrête seul au bout de 60 minutes.
- Parcours retenu : démonstration sur le Space, puis preuves de l’endpoint du 10 novembre
  (rapport de vérification, journal d’audit, trace du cas saisi). L’appel vLLM en direct
  est un bonus, montré à la demande de l’évaluateur si le notebook 09 a pu démarrer.
- Le cas anglais de contrôle part en revue humaine (« delayed emergency ») : c’est le
  garde-fou qui fonctionne, pas une panne.
- Le modèle peut citer un élément absent de la saisie (« hypotension » pour 150/90) :
  je le montre comme une limite connue, pas comme une surprise.
- La démonstration technique ne démontre pas de bénéfice clinique en situation réelle.

## Repli si le temps manque

Je garde les écrans Mission, Données, Résultats, Pipeline, Démo et Décision.
Les paramètres détaillés et les courbes restent disponibles pour les dix minutes
de discussion. Je ne remplace pas la démonstration par la lecture du rapport.
