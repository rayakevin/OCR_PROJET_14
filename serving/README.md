# POC de triage : de l'adaptateur à une interaction traçable

Je retiens le SFT triage v3 après la comparaison avec le pilote DPO. Je construis
une démonstration destinée aux cas fictifs ; je ne présente pas ses propositions
comme une validation clinique.

```text
Questionnaire FR/EN → champs manquants → présentation du cas
                                             ↓
                                  prompt SFT exact → vLLM
                                             ↓
                         ESI/priorité cohérents et sortie complète ?
                             oui : proposition à vérifier
                             non : revue humaine, sans priorité
Chaque interaction → identifiant + date + données + réponse + modèle → SQLite
```

## Organisation

| Fichier | Rôle |
|---|---|
| `domaine.py` | Schéma du cas, questions et lecture prudente de la réponse |
| `backend.py` | Client vLLM, tokenizer du socle, EOS natif et refus de troncature |
| `api.py` | Authentification, parcours, erreurs et consultation de la trace |
| `audit.py` | Journal SQLite, y compris les appels interrompus ou échoués |
| `modele.json` | Révision, hash du LoRA sélectionné, prompts et paramètres |
| `interface.html` | Démonstration locale du questionnaire et de l'audit |
| `lancer_vllm.py` | Contrôle du hash des poids avant le démarrage de vLLM |
| `verifier_endpoint.py` | Trois requêtes fictives : contrat, audit et latence HTTP |

Les questions sont déterministes et dépendent des champs manquants. La mention
lexicale d'une douleur ajoute une question sur son retentissement. Il s'agit
d'une adaptation simple, sans protocole clinique ni détection des négations.
Je ne confie pas la génération des questions au LoRA : il a appris à classer
un cas, pas à conduire un entretien. Les informations inconnues restent telles
quelles ; remplir les champs ne démontre pas que le dossier est suffisant.

Le formulaire produit une présentation différente des cas bruts d'évaluation.
Je dois donc vérifier son effet sur des scénarios séparés. La conservation du
prompt et de l'EOS ne suffit pas à garantir les mêmes sorties entre Transformers
et vLLM : je vérifie cette transition séparément sur trois cas fictifs ; cela ne démontre pas une équivalence des moteurs.

## API et intégration simulée

- `GET /healthz` : processus API actif, sans conclure sur le modèle.
- `GET /readyz` : présence du modèle attendu dans le serveur vLLM.
- `POST /v1/consultations` : nouvelle étape du questionnaire ou proposition.
- `GET /v1/interactions/{uuid}` : trace complète, protégée par la même clé.

Les trois dernières routes nécessitent `Authorization: Bearer …`.
La réponse expose `session_id`, `interaction_id`, `statut`, `esi`, `priorite`,
`validation_humaine_requise` et la provenance du modèle. Ce contrat JSON simule
un point d'intégration SIH ; aucun connecteur hospitalier réel ni échange FHIR
n'est implémenté. La démo ne calcule pas de probabilité de confiance.

Le journal contient les textes fictifs en clair. Le fichier a des droits 0600,
mais il n'est ni chiffré, ni immuable. La clé partagée ne distingue pas les
soignants. Ces limites excluent une revendication de conformité RGPD ou un usage
avec des données de patients. Les logs du serveur vLLM ne doivent pas contenir
les prompts. Une panne de journal bloque la délivrance d'une proposition.

## Vérification sans GPU

```bash
python -m pip install -r serving/requirements.txt
python -m unittest tests.test_serving -v
docker build -f serving/Dockerfile -t ocr14-api:local .
```

Le notebook `04_poc_et_deploiement.ipynb` exécute les vérifications sur CPU et
explique les limites. Les doubles de test sont explicitement identifiés : leurs
résultats ne sont pas des prédictions du modèle. Le test du template emploie un
vrai tokenizer miniature pour détecter une dépendance manquante à Jinja2.

## Reproduire la vérification vLLM sur un environnement GPU

Je récupère depuis Drive le dossier
`experiences/sft_dpo_v3_20261105/sft_triage/adapter`. Son fichier de poids doit
avoir le hash enregistré dans `modele.json`. Je ne remplace pas ce dossier par
un ancien adaptateur local de phase 1.

Je définis `OCR14_API_KEY` et `VLLM_API_KEY` (24 caractères au minimum), ainsi que
`OCR14_ADAPTER_DIR` (chemin absolu du dossier vérifié), dans l'environnement ou
un fichier privé passé à `docker compose --env-file`. Je ne les versionne pas.

```bash
docker compose -f serving/compose.yaml --profile gpu up --build -d
python -m serving.verifier_endpoint --url http://127.0.0.1:8080 \
  --output data/evaluation/poc/verification_endpoint.json
docker compose -f serving/compose.yaml --profile gpu down
```

L'API écoute uniquement en boucle locale ; vLLM n'est pas publié sur l'hôte.
Les volumes conservent l'audit et les caches. Le script de mesure ne teste ni
la charge, ni la pertinence clinique ; la première requête peut inclure le
chargement du tokenizer. Les partitions de test final restent distinctes de ces cas fictifs. Cet essai
porte sur le contrat technique et ne mesure pas la qualité clinique.

Le workflow `.github/workflows/poc.yml` exécute désormais 36 tests et construit Docker.
Sur `main`, il publie ensuite une liste fermée de fichiers dans le Space existant
et vérifie sa révision en fonctionnement. Le secret `HF_TOKEN` reste dans Actions.
La [première exécution complète](https://github.com/rayakevin/OCR_PROJET_14/actions/runs/42074793162)
est réussie. La vérification du serveur vLLM est également exécutée sur une VM Colab temporaire.

## Démonstration gratuite sur Hugging Face

Le 5 novembre 2026, la création du
[Space OCR14](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en) a été
acceptée sur ZeroGPU. Le Space est démarré et deux parcours fictifs FR/EN ont
produit une réponse réelle avec une trace JSON récupérée. Aucun abonnement ni
matériel payant n'a été demandé.
[Conditions et compatibilité ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).

Cette variante utilise Transformers/PEFT et le même LoRA, récupéré depuis Drive
et vérifié par SHA256. Le runtime ZeroGPU emploie PyTorch 2.13.0 ; je ne présume
pas d'une identité numérique avec l'environnement Colab. Je conserve vLLM comme
voie de vérification de l'exigence technique du projet, et je justifierai
explicitement tout écart entre l'hébergement gratuit et cette exigence.
Les traces restent isolées par session et sont téléchargeables en JSON. Le cache
temporaire est nettoyé après une heure ; télécharger la trace assure sa conservation
locale. Cela ne remplace pas un journal hospitalier persistant.

| Fichier dans `space/` | Rôle |
|---|---|
| `parcours.py` | Collecte sur CPU, état par session et trace des erreurs |
| `interface.py` | Formulaire bilingue et export de la trace |
| `app.py` | Chargement du LoRA vérifié ; GPU réservé à la génération |
| `publier.py` | Paquet de fichiers autorisés, sans secrets ni corpus ; création ZeroGPU |
| `requirements.txt` et `README.md` | Dépendances figées et configuration du Space |

Les tests `tests/test_space.py` vérifient sur CPU les deux langues, l'appel final
unique, l'isolation, les entrées invalides, les abstentions et le refus d'autres
poids. La construction de l'interface est également vérifiée avec Gradio 6.31.0.

Deux corrections spécifiques à l'hébergement ont été nécessaires : Pydantic
2.12.5 pour les extensions Gradio ajoutées par Spaces, et lecture du LoRA sur
CPU avant la préparation CUDA de ZeroGPU. Les 17 tests passent aussi avec cette
version de Pydantic.

Sur les deux essais du navigateur, l'étape finale a pris 5,115 s en français
(68 tokens) et 3,870 s en anglais (53 tokens). Cette durée côté serveur inclut
l'appel à ZeroGPU ; ce n'est ni une mesure isolée du calcul GPU, ni une garantie
de latence. Les traces et le commit publié sont conservés dans
`data/evaluation/poc/verification_space_reelle.json` et
`data/deployment/hf_space_v1.publication.json`. Le test final du corpus est mesuré séparément, après sélection sur validation.

Références techniques : [LoRA avec vLLM](https://docs.vllm.ai/en/v0.32.0/features/lora/),
[options du serveur](https://docs.vllm.ai/en/v0.32.0/cli/serve/),
[tests FastAPI](https://fastapi.tiangolo.com/tutorial/testing/).

### Langue et réglages

Je place le sélecteur FR/EN en en-tête : il traduit les textes et commence un nouveau cas.
Le panneau Réglages propose le questionnaire guidé ou la saisie d’un cas complet,
ainsi que l’affichage facultatif des détails techniques. Je conserve les mêmes
paramètres de génération. Le mode direct utilise la présentation saisie sans
ajouter de réponses aux questions manquantes.

Les traductions et changements de mode ont été vérifiés sur le Space. Le nouvel
essai de génération directe a atteint le quota invité ZeroGPU : l’interface
a affiché une indisponibilité, sans attribuer de priorité. Le parcours direct
est testé sur CPU avec un double ; les deux inférences réelles antérieures
restent les preuves du chargement et de la génération par le modèle.

### Vérification ponctuelle sur Colab

Le notebook `evaluation/05_finalisation_colab.ipynb` prépare un environnement
vLLM 0.32.0 séparé de celui de l'évaluation Transformers. `verifier_colab.py`
contrôle le LoRA, démarre vLLM et FastAPI, mesure trois cas fictifs, vérifie les
traces et les refus HTTP, puis arrête les processus. Le notebook sauvegarde les
résultats et synchronise Drive avant de rendre le GPU.

L'exécution du notebook `06_verification_vllm_colab.ipynb` est confirmée sur A100
40 Go. Après le conflit de port 8080, Uvicorn reçoit une socket réservée sur un
port libre. Trois requêtes et leurs traces sont vérifiées : 6,935 s, 0,385 s et
0,514 s. L'anglais produit « delayed emergency », hors contrat : l'API s'abstient
et demande une revue humaine. Les contrôles 401, 422 et cas incomplet passent.
Les appels restent internes à une VM cloud temporaire, sans URL publique persistante. La
démonstration publique ZeroGPU conserve Transformers ; cet écart avec la forme
d'hébergement demandée reste explicite dans le rapport.
