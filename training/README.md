# Entraînement LoRA

Le script `entrainer_lora.py` entraîne le même adaptateur en supervisant uniquement les tokens de la réponse `assistant`. Il sauvegarde l'adaptateur dans `models/` et trace les paramètres et les métriques dans MLflow.

```bash
uv run python training/entrainer_lora.py \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --output models/phase1_qa_eos_v2 --max-length 5120
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Avant le run complet, un pilote limite volontairement les données et le nombre de pas.

## Correction du protocole SFT

Je partage le prompt avec l'inférence dans `format_sft.py`, y compris les balises
de réflexion vide du tokenizer. Seule la réponse contribue à la loss ; elle se
termine par l'EOS natif du socle. Je refuse les exemples trop longs plutôt que
de couper leur réponse ou leur cas. Les jeux QA actuels tiennent dans 5 120 tokens.

Le notebook `evaluation/03_execution_colab.ipynb` conserve les résultats initiaux,
explique les défauts constatés et présente le pilote corrigé. Les nouveaux runs
utilisent un autre dossier. Le protocole sauvegarde les identifiants sélectionnés,
les empreintes, les versions, la révision du socle et les paramètres MLflow.

Pour la seconde phase, `--adapter` poursuit l'adaptateur corrigé, avec
`--phase phase2_triage` et les chemins `--train` / `--validation` correspondants.
Cette commande ne doit pas réutiliser l'ancien adaptateur au format incohérent.
L'inférence corrigée utilise `--stop-token eos` et enregistre la cause d'arrêt.

## Essai de rappel QA en phase 2

Après la baisse des scores QA, je teste dans la section 12 du notebook un mélange
de 1 900 triages et 212 QA du train (106 par langue), tirées avec la seed 3407.
Je repars de la phase 1 corrigée : 357 étapes, batch effectif 16, learning rate
0,0001, `--warmup-steps 20` et checkpoints tous les 120 pas. La limite de contexte
passe à 5 120 pour conserver les réponses QA entières. Les tokens de réponse QA
représentent environ 28 % du mélange, malgré seulement 10 % des exemples.

L'audit a détecté un doublon de prompt entre train et validation triage sous des
identifiants différents. J'exclus ses deux versions linguistiques du nouvel
entraînement, sans réécrire les partitions historiques. Je compare les scores
sur les 238 entrées puis sur les 236 entrées sans la paire contaminée. Un autre
doublon validation/test est ensuite écarté des partitions v3. Le notebook
conserve les identifiants et les empreintes de cet audit.

Le contrôle préliminaire du DPO relève aussi quatre prompts du train DPO présents
dans la validation ou le test QA. Ils sont retirés dans les partitions v3 avant le DPO. Je vérifie l'adéquation des préférences au triage ; les scores SFT ne justifient
pas à eux seuls le lancement d'un DPO généraliste.

Les calculs GPU sont exécutés sur Colab. Les résultats, les adaptateurs et le suivi
sont sauvegardés dans Drive avant de libérer l’environnement.

## Partitions v3 et pilote DPO

Je reconstruis la version nettoyée avec `data_ingestion/figer_partitions.py`.
Les fichiers historiques restent disponibles ; les nouvelles partitions, leurs
empreintes et les exclusions sont dans `data/processed/partitions_v3`.
Le train triage contient 1 888 entrées, la validation 238 et le test figé
226. La QA reste inchangée. Les exclusions portent toujours sur la paire de langues.

Je reprends seulement le SFT de triage avec 357 pas et les mêmes paramètres.
`training/preparer_dpo.py` construit ensuite un pilote de 256 paires de train et
64 de validation, stratifié par catégorie de préférence. Une seule paire du
train dépasse 4 096 tokens : elle est écartée de l'éligibilité, jamais tronquée.

`training/entrainer_dpo.py` utilise TRL 1.17.1 pendant 32 pas. La référence est une
copie figée du LoRA SFT initial ; désactiver ce LoRA reviendrait à utiliser le socle
comme référence et changerait l'expérience. Je vérifie les empreintes des poids,
les frontières des tokens et l'EOS. Un test miniature CPU vérifie la mise à jour
de la politique et l'immobilité de la référence avant les calculs Colab.

Le notebook 03 intègre les scripts, les commandes et les résultats au fil de leur
obtention. Un pilote DPO valide un pipeline et donne un premier signal ; il ne
constitue ni un entraînement complet ni une validation clinique.

Le pilote exécuté sur Colab améliore la perte DPO de validation (0,6931 →
0,5361), mais pas la macro-F1 des priorités (0,7715 → 0,7674). Les cinq
sous-triages critiques persistent. À cette étape, je conserve le SFT v3 comme candidat
principal du POC et le DPO comme variante expérimentale. Le SFT mixte lui succède ensuite. La section 19 du
notebook détaille aussi l'écart entre langues et la perte de capacité QA.
Les résultats et les 1 000 prédictions comparatives sont disponibles dans
`data/evaluation/colab_sft_dpo_v3`, en complément de la sauvegarde Drive.

## SFT mixte puis DPO ciblé (6 novembre)

Le notebook 07 part du socle avec un nouveau LoRA sur l'union QA/triage. L'époque 3
est retenue et déployée depuis le 10 novembre ; les sept époques supplémentaires,
avec un nouvel optimiseur et un taux réduit, n'apportent pas de gain global.

Le notebook 08 prépare une autre intervention : `preparer_dpo_triage.py` transforme
les erreurs réelles du SFT mixte sur le **train triage uniquement** en préférences.
La référence silver reste inchangée ; la réponse rejetée est la génération effective.
Les sorties invalides et les troncatures sont exclues. Je refuse un pilote sans
sous-triages critiques et sur-triages, plutôt que d'inventer des erreurs.

Le contrôle interne DPO représente 20 % des groupes de cas, FR/EN indissociables.
Ces cas ont déjà été vus en SFT : cette loss ne mesure pas la généralisation.
Le pilote poursuit l'époque 3 pendant 32 pas, LR 2e-6, beta 0,1, microbatch 1,
accumulation 16. Je mesure séparément QA et triage aux pas 16 et 32.
Le script conserve aussi son ancien comportement par défaut pour les expériences historiques.

Je vérifie la référence figée, les SHA256, le format et les frontières de tokens.
Les tests CPU vérifient le pipeline ; ils ne préjugent pas des scores obtenus sur GPU.
Le checkpoint sauvegardé dans `adapter/` suit la loss de préférence : le choix
métier repose sur les deux checkpoints explicitement évalués, pas sur ce nom de dossier.


## Décision après le DPO ciblé

Le run du 6 novembre est terminé : 355/85 paires, 32 pas. Sur validation,
le rappel urgent passe de 91 % pour le SFT mixte réévalué à 87 % puis 83 %,
et la macro-F1 de 0,7875 à 0,7457 puis 0,7373. Je ne retiens aucun de ces DPO.
Les 1 500 prédictions, six rapports, poids et états d’entraînement sont récupérés ;
95 empreintes correspondent. Le notebook 08 conserve le bilan et les limites.
