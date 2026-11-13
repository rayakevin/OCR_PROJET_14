# Prototype de triage médical bilingue
## Rapport technique de mission

**Kevin Raya · OCR Projet 14 · 13 novembre 2026**

## Synthèse et recommandation

La mission évalue la faisabilité d’une aide au triage français / anglais : recueillir une présentation clinique, produire une proposition de priorité et conserver une trace exploitable par un soignant. Le prototype associe un modèle Qwen3-1.7B-Base spécialisé par LoRA à un questionnaire, des contrôles de sortie et un service d’inférence.

**La faisabilité technique est démontrée ; l’utilité et la sécurité cliniques restent à établir.** Le système fonctionne de bout en bout, y compris via une API vLLM sur GPU distant. L’apprentissage améliore fortement le respect du format et le classement par rapport au socle. Il ne supprime ni les urgences manquées ni les justifications infidèles.

| Question de mission | Conclusion |
|---|---|
| Un petit modèle peut-il apprendre la tâche ? | Oui. La spécialisation produit des classements structurés avec environ 1 % de paramètres entraînés. |
| Peut-il conserver QA et triage ? | Le SFT mixte préserve mieux les deux capacités que le SFT successif, au prix d’un rappel urgent inférieur. |
| Le DPO corrige-t-il les erreurs importantes ? | Pas dans les essais réalisés. Le DPO ciblé diminue le sur-triage mais augmente les urgences manquées. |
| Le service est-il déployable ? | Oui pour une démonstration : Space public et API temporaire authentifiée sur Colab. |
| Peut-il être utilisé auprès de patients ? | Les résultats ne justifient pas cet usage : références non validées cliniquement, sous-triage et hallucinations persistants. |

### Solution retenue

Le candidat de démonstration est le **SFT mixte, époque 3**. Sur validation, il atteint 0,7874 de macro-F1 priorité, retrouve 91 urgences sur 100 et répond correctement à 57,25 % des QCM. Il conserve davantage de QA que le SFT successif, qui retrouve toutefois 95 urgences sur 100. Ce choix répond à l’objectif expérimental à deux tâches ; il ne constitue pas un arbitrage clinique validé.

Les priorités proposées regroupent ESI 1–2 en « maximale », ESI 3 en « modérée » et ESI 4–5 en « différée ». Le soignant garde la décision. Le gain de temps, la qualité de prise en charge et l’effet sur les patients n’ont pas été mesurés.

### Enseignements déterminants

**La qualité de la référence précède la sophistication du modèle.** Un consensus de LLM fournit un corpus exploitable pour expérimenter, pas une vérité clinique. **L’objectif d’apprentissage ne suffit pas à choisir le modèle.** Une loss plus faible peut accompagner une dégradation du triage. **Le mélange des tâches limite l’oubli, sans supprimer le compromis.** La prochaine étape utile est donc une évaluation clinique indépendante, accompagnée de comparaisons d’architectures ciblées.

<!-- page -->
## 1. Données et fiabilité de l’évaluation

### Construire une cible qui n’existait pas dans les sources

Les corpus disponibles contiennent des questions médicales, des vignettes ou des préférences entre réponses, mais pas directement la cible ESI du projet. La préparation transforme ces contenus en deux tâches distinctes : répondre à une question médicale et classer une présentation de patient.

| Source | Rôle dans le corpus | Limite principale |
|---|---|---|
| MediQAl, français | QA et vignettes candidates au triage | Cas pédagogiques ; licence CC BY 4.0 documentée |
| UltraMedical-Preference, anglais | Préférences DPO et candidats au triage | Préférence native différente d’une validation ESI ; licence MIT documentée |
| MedQuAD, anglais | Questions ouvertes avec réponses rédigées | Droits d’usage à clarifier pour la diffusion envisagée |
| FrenchMedMCQA | Source explorée, non retenue | Recouvrement et licence non renseignée lors de l’examen |

Sur 4 007 candidats au triage, 1 189 cas uniques sont retenus : 700 MediQAl et 489 UltraMedical. La sélection exige une qualification compatible avec le premier accueil, puis un consensus sur la priorité et l’ESI exact. Une majorité sur la priorité seule ne suffit pas à produire une cible ESI. Codex et Claude annotent dans des sessions d’agent ; Mistral intervient par API, sur un sous-ensemble ciblé de 1 516 candidats pour UltraMedical. Les contrôles automatiques vérifient le format, les citations et la cohérence des annotations, pas leur justesse médicale.

Les références sont donc **silver**, issues de modèles et sans relecture clinique indépendante. Le consensus réduit certains désaccords ; il peut aussi conserver des biais partagés et écarter les situations les plus ambiguës. Les traductions augmentent la couverture linguistique, pas le nombre de cas indépendants.

### Partitions finales et prévention des fuites

| Corpus | Entraînement | Validation | Test | Total |
|---|---|---|---|---|
| QA | 2 098 | 262 | 262 | 2 622 |
| Triage | 1 888 | 238 | 226 | 2 352 |
| Ensemble SFT | 3 986 | 500 | 488 | 4 974 |

Le corpus initial de 5 000 entrées est réduit après audit des recouvrements et retrait des cas de test exposés dans un pilote. Les versions française et anglaise d’un cas restent dans la même partition. Les identifiants, empreintes et exclusions rendent cette séparation contrôlable. Ces mesures limitent les fuites détectables ; elles n’excluent pas toute proximité sémantique ni une exposition préalable du socle.

### Protection des données

Le projet utilise des corpus publics ; la démonstration et ses journaux sont réservés aux cas fictifs. Un audit Presidio des 10 000 messages initiaux signale surtout des lieux, dates et noms dont une partie correspond à des éponymes médicaux. L’examen d’un échantillon motive l’absence de masquage automatique systématique, qui aurait altéré le contenu médical. Cet audit ne prouve ni une anonymisation complète ni une conformité pour un usage réel. Les droits de réutilisation, la revue des données et le cadre de traitement restent à sécuriser avant toute extension.

<!-- page -->
## 2. Stratégie d’apprentissage et choix du candidat

### Une adaptation légère, avec un protocole commun

Qwen3-1.7B-Base constitue le socle du POC. LoRA permet d’en conserver les poids figés et d’apprendre 17,4 millions de paramètres supplémentaires, soit environ 1 %. Le rang est de 16 et l’alpha de 32. Cette approche rend les essais et la distribution des adaptateurs relativement légers ; elle ne garantit pas la conservation des capacités entre tâches.

Le SFT supervise uniquement la réponse attendue de l’assistant. Un format commun à l’entraînement et à l’inférence, l’EOS natif et le refus des troncatures silencieuses corrigent les défauts de fin de réponse observés au départ. L’enseignement est opérationnel : un résultat médiocre peut venir du protocole de préparation, pas seulement des capacités du socle.

| Démarche | Hypothèse | Résultat et décision |
|---|---|---|
| SFT successif : QA puis triage sur le même adaptateur | Apprendre le format général puis spécialiser | Bon classement, mais forte perte de QA ; conservé comme comparaison |
| Rappel partiel de QA pendant la spécialisation | Limiter l’oubli sans changer toute la recette | Compromis insuffisant pour remplacer le candidat ; conduit à tester un mélange complet |
| SFT mixte depuis le socle | Apprendre conjointement QA et triage | Meilleur équilibre entre tâches ; époque 3 retenue pour le POC |
| Poursuite du mixte jusqu’à dix époques cumulées | Améliorer le classement avec davantage d’apprentissage | Aucun candidat ne domine les critères ; poursuite non retenue |

### Comparaison sur la même validation

| Candidat | QCM corrects | F1 lexical QA | Macro-F1 priorité | Urgents retrouvés |
|---|---|---|---|---|
| SFT successif v3 | 3,82 % | 0,0359 | 0,7715 | 95/100 |
| Mixte, époque 1 | 53,44 % | 0,3364 | 0,6680 | 76/100 |
| Mixte, époque 2 | 58,02 % | 0,3571 | 0,7720 | 85/100 |
| Mixte, époque 3 | 57,25 % | 0,4341 | 0,7874 | 91/100 |

Le mixte entraîne les 3 986 exemples avec un nouveau LoRA, un batch effectif de 16, un taux maximal de 0,0001 et trois époques. Les sauvegardes 250, 500 et 750 correspondent aux fins d’époque. Le contexte maximal est de 5 120 tokens, la précision BF16 et la seed 3407. La QA représente environ 53 % des exemples mais 79 % des tokens supervisés : équilibrer les volumes ne revient pas à équilibrer les contributions à l’apprentissage.

La poursuite repart des poids de l’époque 3 avec un taux réduit, un optimiseur et un calendrier réinitialisés. L’époque 5 améliore la macro-F1 à 0,8050 mais ne retrouve que 86 urgences sur 100 ; l’époque 6 en retrouve 92, avec une macro-F1 de 0,7711. **Plus d’époques ne fournit pas ici un meilleur compromis démontré.** Cela ne prouve pas que toute poursuite serait inutile.

Le choix du mixte ne masque pas sa contrepartie : quatre urgences supplémentaires sont manquées par rapport au successif. La conservation de QA justifie le candidat de démonstration, pas une autorisation d’usage clinique.

<!-- page -->
## 3. Alignement DPO : résultat négatif, enseignement utile

Le DPO apprend à favoriser une réponse par rapport à une autre pour une même demande, avec une référence figée. Il modifie la préférence du modèle ; il ne transforme pas automatiquement les références en décisions médicales fiables.

### Deux hypothèses testées

**Préférences générales.** Un pilote de 32 pas utilise une sélection de 256 paires d’entraînement et 64 de validation, issues d’UltraMedical nettoyé. La loss baisse de 0,6931 à 0,5361, mais les cinq sous-triages critiques du SFT successif restent présents. L’amélioration de l’objectif DPO ne se transfère pas au besoin métier. Ce pilote n’est pas retenu.

**Préférences ciblées sur les erreurs ESI.** Les réponses du mixte sur le train triage servent à construire 355 paires d’apprentissage et 85 de contrôle interne. Les cas de validation métier ne servent pas à fabriquer ces paires. La comparaison utilise 32 pas, un taux de 0,000002, beta 0,1 et un batch effectif de 16. La référence SFT est réévaluée dans la même session pour comparer les variantes.

| Validation du DPO ciblé | Référence mixte | DPO, 16 pas | DPO, 32 pas |
|---|---|---|---|
| Urgents retrouvés | 91/100 | 87/100 | 83/100 |
| Macro-F1 priorité | 0,7875 | 0,7457 | 0,7373 |
| Sur-triage | 18,49 % | 14,71 % | 12,18 % |
| QCM corrects | 56,49 % | 60,31 % | 58,78 % |

Les légers écarts entre cette référence et les scores initiaux du mixte proviennent de la réévaluation ; ils ne correspondent pas à un nouvel entraînement SFT.

### Interprétation et conséquence

Le DPO ciblé **réduit le sur-triage au prix d’une dégradation du rappel urgent et de la macro-F1**. Il échoue au critère expérimental recherché : améliorer le rappel tout en limitant les pertes sur les autres capacités. Aucun des deux checkpoints n’est retenu pour la démonstration.

Les paires contiennent 234 corrections de sur-triage, 72 de sous-triage critique et 49 autres erreurs. Cette répartition peut favoriser une baisse de l’urgence prédite, mais l’essai n’isole pas cette cause. Les différences de formulation entre réponses préférées et rejetées peuvent également orienter l’apprentissage.

**La suite pertinente serait de revoir le signal de préférence, pas simplement d’allonger le calcul.** Des paires équilibrées par gravité, relues par des soignants, permettraient de tester cette hypothèse. Elles devraient être comparées sur les mêmes cas indépendants, en conservant le rappel des urgences comme critère de sécurité expérimental.

Les résultats du pilote généraliste sont conservés, mais ses poids finaux ne sont pas disponibles dans l’archive récupérée. Les adaptateurs du DPO ciblé sont conservés. Cette différence limite la reproductibilité du premier essai, sans modifier sa conclusion observée.

<!-- page -->
## 4. Ce que les évaluations établissent réellement

### Des métriques complémentaires

| Mesure | Question couverte | Limite d’interprétation |
|---|---|---|
| Rappel urgent | Parmi les ESI 1–2 de référence, combien sont reconnus urgents ? | Une bonne moyenne ne supprime pas les erreurs graves |
| Macro-F1 priorité | Les trois priorités sont-elles correctement distinguées ? | Peut progresser alors que le rappel urgent recule |
| Accuracy ESI et taux analysable | Le niveau exact et le format sont-ils retrouvés ? | Un format correct ne garantit pas la justesse |
| Sous-triage / sur-triage | Dans quel sens se produisent les erreurs ? | Dénominateurs à distinguer du rappel urgent |
| QCM / F1 lexical | La capacité QA est-elle conservée ? | Le recouvrement lexical ne valide pas le contenu médical |
| Accord FR/EN | Une traduction conserve-t-elle le classement ? | Deux réponses identiques peuvent être fausses |

La validation contient 238 entrées de triage, correspondant à 119 cas bilingues, et 262 QA. Les petites différences entre variantes ne sont pas accompagnées d’une démonstration de significativité. La QA confond partiellement langue et tâche : QCM français et questions ouvertes anglaises. Elle ne permet pas d’isoler un effet de langue.

### Test indépendant : portée précise

Le test final compare le socle au **SFT successif**, sélectionné à ce moment. Il comprend 226 triages et 262 QA par modèle. Les prédictions ont été conservées et les métriques recalculées localement.

| Test final | Socle | SFT successif |
|---|---|---|
| Macro-F1 priorité | 0,0556 | 0,7858 |
| ESI exact | 3,10 % | 74,78 % |
| ESI analysables | 10,62 % | 100 % |
| Urgents retrouvés | 0/98 | 88/98 |

Ce gain inclut l’apprentissage du format ; il ne mesure pas uniquement une amélioration du raisonnement médical. Le test confirme aussi l’oubli QA : aucun QCM correct pour le successif sur ce jeu.

**Le mixte déployé est choisi sur validation ; il ne dispose pas encore d’une confirmation indépendante.** Les scores du test successif ne lui sont pas attribuables. Une nouvelle référence est nécessaire, le test existant ayant déjà participé à l’analyse du projet.

### Limites qui conditionnent toute suite clinique

Le mixte manque neuf entrées urgentes sur cent en validation, concernant sept cas distincts. Des justifications inventent ou contredisent des éléments : « hypotension » apparaît pour une tension saisie à 150/90, y compris avec les deux moteurs d’inférence. Changer le moteur de service ne corrige donc pas ce défaut observé.

La référence silver, les cas pédagogiques et l’exclusion de situations ambiguës limitent la représentativité. Les contrôles applicatifs détectent des sorties invalides ou incohérentes, pas toutes les erreurs médicales. La validation humaine est une condition du parcours ; son efficacité comme mesure de réduction du risque n’a pas été évaluée.

<!-- page -->
## 5. Architecture, déploiement et exploitation

### Une séparation explicite des responsabilités

Le questionnaire détermine les informations manquantes ; le modèle propose un classement ; le service contrôle la sortie et conserve une trace. Le questionnaire est déterministe : le POC ne contient pas encore un agent entraîné à conduire librement un entretien clinique. Aucun connecteur SIH opérationnel n’est implémenté.

| Composant | Choix technique | Responsabilité |
|---|---|---|
| Apprentissage | PyTorch, Transformers, PEFT, TRL | SFT et DPO ; adaptateurs séparés du socle |
| Suivi | MLflow, fichiers de prédictions, empreintes | Paramètres, métriques et traçabilité des versions |
| Démonstration | Gradio, Transformers/PEFT, ZeroGPU | Parcours bilingue public et export de session |
| API | FastAPI, Pydantic, vLLM | Contrat typé et inférence sur GPU distant |
| Audit | SQLite et identifiants de requête | Résultats métier et événements HTTP |
| Livraison | Docker, GitHub Actions | Tests, construction de l’image et publication du Space |

### Mise en service vLLM

L’API a été mise en service sur un GPU Colab distant, exposée par un tunnel HTTPS et vérifiée par des appels sur cas fictifs. Le notebook 06 automatise la préparation, le chargement du modèle et de l’adaptateur, les contrôles de disponibilité, la sauvegarde des traces et l’arrêt. L’URL est temporaire ; il s’agit d’un déploiement de démonstration, pas d’un service permanent.

vLLM garde le modèle chargé, gère les requêtes concurrentes et optimise la mémoire d’attention. Ce choix vise le débit et l’exploitation de l’inférence. Il ne modifie pas la qualité clinique du modèle. Le Space utilise un autre moteur, Transformers/PEFT, avec allocation ZeroGPU à la demande.

### Contrôles et observabilité

Les consultations et les traces API sont protégées par une clé. Les entrées sont validées avant génération. Une sortie inexploitable entraîne une revue humaine ; une indisponibilité du moteur est signalée, et une panne du journal bloque la délivrance d’une proposition. Le manifeste identifie le socle et l’adaptateur retenu, dont l’empreinte est vérifiée.

L’audit métier conserve demande, résultat, modèle et statut pour les cas fictifs. Les événements HTTP enregistrent route, durée, statut et type d’erreur, avec un identifiant de corrélation, sans clé ni corps dans ces logs techniques. MLflow suit les expériences ; il ne remplace pas le journal du service. Le Space conserve un état de session et un export distincts de l’audit SQLite de l’API.

### Livraison continue

GitHub Actions exécute les 52 tests, construit l’image Docker, puis publie le Space après succès sur main. Les tests couvrent notamment extraction des réponses, métriques, questionnaire, authentification, audit et erreurs. Le déploiement vLLM reste lancé à la demande par Colab : **la CI/CD du Space ne constitue pas un déploiement automatique de l’API GPU**. Ces vérifications établissent un comportement logiciel, pas une validation médicale.

<!-- page -->
## 6. Performance, ressources et passage à l’échelle

### Mesures utiles à l’exploitation

| Observation | Résultat | Portée |
|---|---|---|
| API publique vLLM, L4, 10 novembre | Médiane de 1,229 s sur trois requêtes ; premier appel de 7,74 s | Latence observée, avec effet de premier appel |
| Même API, huit requêtes simultanées | 1,537 s au total ; 5,21 requêtes/s ; 329 tokens/s | Petit essai de concurrence, pas une charge soutenue |
| Démonstration ZeroGPU | Environ 3,9 à 5,1 s sur les propositions observées | Allocation GPU et moteur différents ; comparaison non contrôlée |
| SFT mixte | Environ 16 min 25 s, évaluations comprises | Mesure de session, pas un benchmark matériel |
| Cinq calculs principaux SFT/DPO | Environ 1 h 21 de GPU cumulé | Exclut installations, phase QA et essais supplémentaires |

Le modèle de 1,7 milliard de paramètres limite la taille des poids, mais la mémoire nécessaire dépend aussi du contexte, des activations et du nombre de requêtes. Un dépassement mémoire lors d’un essai d’entraînement rappelle que la longueur des réponses QA compte autant que la taille du socle pour dimensionner le calcul.

### Lecture économique

Le coût du POC provient d’exécutions GPU ponctuelles. Un service permanent change l’équation : les heures de disponibilité, la supervision et la protection des données dominent rapidement le coût de l’adaptation LoRA. Les tarifs Colab dépendent du compte et des unités consommées ; aucun coût total audité n’est établi ici.

Pour comparer les options, le budget d’hébergement doit être calculé à partir du tarif GPU effectivement proposé et des heures actives, puis complété par stockage, supervision et exploitation. À titre de simulation budgétaire uniquement, un tarif de 0,80 $/h représenterait 576 $ pour 720 heures, avant ces autres postes. Ce montant n’est ni un tarif contractuel vérifié ni un devis.

La L4 est le matériel des mesures rapportées, pas une contrainte de la démonstration. Un autre GPU peut être utilisé avec un environnement compatible ; les performances doivent alors être mesurées de nouveau.

### Ordre recommandé pour la suite

| Priorité | Travail attendu | Décision rendue possible |
|---|---|---|
| 1. Référence clinique | Cas relus, représentatifs, urgences rares et ambiguïtés incluses | Définir des critères d’acceptation avec les soignants |
| 2. Évaluation indépendante | Comparer candidats et erreurs graves sur des cas nouveaux | Confirmer ou rejeter le compromis du mixte |
| 3. Service durable | Charge prolongée, pannes, sécurité, conservation et accès | Dimensionner un pilote et son exploitation |
| 4. Pilote encadré | Mesurer l’usage réel, la charge du soignant et les erreurs | Établir une valeur clinique, au-delà des scores |

Le cadre de protection des données, l’hébergement approprié, les responsabilités et les modalités de supervision doivent être définis avec les acteurs concernés avant un pilote sur données de patients. La recommandation actuelle reste une démonstration sur cas fictifs, sans automatisation de décision clinique.

<!-- page -->
## 7. Pistes d’exploration au-delà du POC

LoRA est une méthode d’adaptation économique, pas une architecture clinique complète. Les ouvertures suivantes reposent sur des travaux publiés ; leur transposition au projet reste une hypothèse à évaluer. Cette recherche ciblée n’est ni une revue systématique ni une qualification de solutions commerciales.

### Classification hybride plutôt que génération seule

Lafuente et Rahim (2025) comparent des LLM généralistes, spécialisés et ajustés à des approches de machine learning sur 1 000 dossiers équilibrés issus de MIMIC-IV-ED. Les modèles classiques enrichis par des caractéristiques issues de LLM ou de graphes de connaissances dépassent les LLM seuls ; les meilleurs F1 moyens restent inférieurs à 0,60 dans ce protocole [1]. Ces valeurs ne sont pas comparables directement à celles de notre corpus silver et de nos trois priorités.

**Expérience proposée :** utiliser le langage pour extraire des faits, puis comparer un classifieur supervisé et un moteur de règles défini avec les soignants à la génération ESI actuelle. L’objectif serait de mieux maîtriser le classement et son incertitude, sans supposer qu’une réponse textuelle est nécessaire à toutes les étapes.

### Entretien et triage comme spécialistes distincts

AMIE étudie l’apprentissage d’entretiens médicaux avec simulation et feedback. Son évaluation de 2025 confronte le système à des médecins sur 159 scénarios avec patients simulés et examine plusieurs dimensions du dialogue [2]. Ce travail soutient l’intérêt d’une évaluation spécifique de la collecte ; il ne valide ni notre modèle ni une chaîne autonome de triage.

**Expérience proposée :** un modèle de collecte pose les questions et transmet une synthèse factuelle à un modèle de triage, éventuellement via deux adaptateurs sur un socle commun. Notre QA actuelle ne suffit pas à former le premier rôle. Les omissions, inventions, négations et incertitudes doivent être contrôlées dans la transmission. Un diagnostic supposé ne doit pas être présenté comme un fait au second modèle. L’architecture sera comparée au modèle unique en précision, latence et propagation des erreurs.

### Recherche documentaire pour étayer les explications

MIRAGE/MedRAG (2024) évalue le RAG sur 7 663 questions médicales et 41 configurations. L’étude rapporte des améliorations de QA selon les corpus, moteurs de recherche et modèles [3]. Ce résultat concerne des benchmarks de questions-réponses, pas une validation du triage aux urgences.

**Expérience proposée :** fournir des recommandations sélectionnées et versionnées pour soutenir les explications, avec citations et contrôle de fidélité. Le RAG ne garantit ni un classement correct ni l’absence d’hallucination ; les faits du patient doivent rester séparés des documents retrouvés.

### Comparer les méthodes sur une même référence

Un modèle plus grand, un ajustement complet des poids ou des préférences mieux équilibrées peuvent entrer dans la comparaison. Aucun n’est présumé supérieur. La priorité serait une référence commune et des essais où un changement est isolé : données, architecture ou apprentissage. Les critères couvriraient rappel urgent, erreurs par gravité, fidélité, abstention, coût et temps de réponse. Cela permettrait de choisir une approche sur sa contribution mesurée au besoin, plutôt que sur sa complexité.

<!-- page -->
## Références et accès aux livrables

### Études mobilisées pour les perspectives

[1] Lafuente, C. et Rahim, M. (2025). Evaluating large language models on hospital health data for automated emergency triage. International Journal of Computer Assisted Radiology and Surgery. DOI : 10.1007/s11548-025-03475-1. Texte et résumé institutionnels : https://riunet.upv.es/handle/10251/224376

[2] Tu, T. et al. (2025). Towards conversational diagnostic artificial intelligence. Nature, 642, 442–450. DOI : 10.1038/s41586-025-08866-7. https://www.nature.com/articles/s41586-025-08866-7

[3] Xiong, G., Jin, Q., Lu, Z. et Zhang, A. (2024). Benchmarking Retrieval-Augmented Generation for Medicine. Findings of ACL, 6233–6251. DOI : 10.18653/v1/2024.findings-acl.372. https://aclanthology.org/2024.findings-acl.372/

### Parcours technique actuel : six notebooks

| Emplacement | Contenu |
|---|---|
| notebooks/01_preparation_donnees.ipynb | Sources, annotation, consensus, formats et partitions |
| notebooks/02_evaluation.ipynb | Métriques, protocole et portée des évaluations |
| notebooks/03_sft.ipynb | Comparaison successif / mixte et sélection |
| notebooks/04_dpo.ipynb | Préférences, résultats et motifs de rejet |
| notebooks/05_deploiement.ipynb | Architecture, contrôles, CI/CD et résultats de service |
| notebooks/06_lancer_demo_colab.ipynb | Lancement de la démonstration vLLM sur GPU distant |

Les carnets 02 à 05 relisent les bilans existants sans relancer l’entraînement. Le carnet 06 est opérationnel et requiert une session Colab adaptée. Les journaux détaillés d’essais restent dans les archives locales ; ils ne constituent plus le parcours principal de lecture.

### Code, données et modèles

Dépôt public : https://github.com/rayakevin/OCR_PROJET_14

Démonstration : https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en

Dataset versionné : https://huggingface.co/datasets/rayakevin/ocr14-medical-bilingue-v3

Adaptateurs : https://huggingface.co/rayakevin/ocr14-triage-lora-experiences

Les deux dépôts de données et de modèles sont privés ; leur accès nécessite une autorisation. Les fichiers nécessaires figurent dans la remise. Les versions sont référencées dans `reports/publication_livrables_hf.json` et le modèle servi dans `serving/modele.json`. Les résultats agrégés sont consultables dans les notebooks et la page explicative ; les prédictions et archives permettent une analyse plus détaillée.

### Provenance et reproductibilité

Les annotations proviennent de LLM. La préparation du code et de la documentation est assistée par des outils d’IA ; les contrôles automatiques ne sont pas présentés comme une relecture clinique. Paramètres, seed, empreintes des données et identités des adaptateurs sont conservés. Une seed commune ne garantit pas une identité numérique entre moteurs, matériels ou versions de bibliothèques.

Ce rapport synthétise les résultats acquis et les recommandations de suite. Les pistes de la section 7 n’ont pas été implémentées ni évaluées dans le POC.
