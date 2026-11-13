Le [carnet Déploiement](../notebooks/05_deploiement.ipynb) présente la synthèse du service et de ses vérifications.

# POC de triage : de l'adaptateur à une interaction traçable

Je sers le SFT mixte époque 3, retenu après la comparaison avec le SFT successif v3
et les deux essais DPO ; `modele.json` fixe sa révision et son empreinte. Je construis
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

## Vérification et déploiement

Depuis la racine :

```bash
python -m pip install -r serving/requirements.txt pytest scikit-learn
python -m pytest tests/test_serving.py tests/test_space.py -v
docker build -f serving/Dockerfile -t ocr14-api:local .
```

Les tests utilisent des doubles sur CPU ; ils ne génèrent pas de réponses du modèle.
Le workflow `.github/workflows/poc.yml` exécute la suite complète et construit
l'image, puis publie le Space après succès sur main. Le secret `HF_TOKEN` reste
dans Actions. L'API GPU n'est pas déployée automatiquement par ce workflow.

## Space public

Le [Space OCR14](https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en)
utilise Transformers/PEFT et ZeroGPU avec le SFT mixte époque 3. Les traces sont
isolées par session et exportables en JSON ; elles ne remplacent pas un journal
persistant. Les quotas et l'attribution du GPU influencent la latence.

| Fichier dans `space/` | Rôle |
|---|---|
| `parcours.py` | Questionnaire et état de session |
| `interface.py` | Formulaire bilingue et export |
| `app.py` | Chargement du LoRA et génération ZeroGPU |
| `publier.py`, `deployer.py` | Paquet autorisé, publication et contrôle du Space |
| `support.py`, `support/` | Page explicative autonome : HTML, CSS, JavaScript et résultats agrégés |

La page explicative comporte cinq chapitres : vue d'ensemble, données, SFT, DPO
et service. Ses simulations sont locales, sans appel au modèle. Le générateur
intègre styles, script et résultats dans un HTML autonome également fourni sous
`output/Page_explicative_projet.html`.

## Endpoint vLLM sur GPU distant

Le [notebook 06](../notebooks/06_lancer_demo_colab.ipynb) démarre vLLM et FastAPI,
expose une URL HTTPS temporaire, vérifie les appels et sauvegarde les traces.
Il nécessite un GPU compatible et le paquet `ocr14_endpoint_soutenance_20261110.zip`,
fourni dans la remise, dans `MyDrive/OCR_PROJET_14/`. La L4 a servi aux mesures ;
elle n'est pas imposée pour une nouvelle démonstration.

Activer `LANCER_ENDPOINT`, saisir une clé temporaire d'au moins 24 caractères
et attendre la vérification HTTPS avant d'utiliser `DEMO_URL`. La cellule reste
active pendant le service ; l'arrêt automatique intervient après 60 minutes.
Les secrets ne doivent pas être copiés dans le notebook sauvegardé.

La vérification publique du 10 novembre observe une médiane de 1,23 s sur trois
appels et huit requêtes simultanées en 1,54 s. Ce petit essai ne démontre pas la
tenue en charge prolongée. Le Space et cet endpoint sont deux voies distinctes.

Références : [LoRA avec vLLM](https://docs.vllm.ai/en/v0.32.0/features/lora/),
[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).
