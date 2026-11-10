---
title: OCR14 — Triage FR EN
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
tags:
  - medical
  - educational
  - lora
  - zerogpu
---

# Démonstration pédagogique de triage bilingue

Je présente un questionnaire adaptatif simple, suivi d'une proposition ESI et
d'une priorité. **Cas fictifs uniquement. Aucune décision clinique autonome.**
Every proposal requires professional review. Fictional cases only.

## Modèle et limites

Je réutilise Qwen/Qwen3-1.7B-Base et le LoRA SFT mixte époque 3, entraîné en une
seule fois sur la QA médicale et le triage, puis sélectionné sur la validation.
`serving/modele.json` fixe la révision du socle, l'empreinte des poids,
les prompts, l'EOS natif et les limites de tokens. Les annotations d'apprentissage
proviennent d'un consensus de LLM ; elles ne constituent pas une validation clinique.

Sur la validation : macro-F1 des priorités 0,7874, rappel des urgences maximales
0,91, soit 9 erreurs critiques sur 100 entrées urgentes, et 57 % de QCM corrects
en QA. Ces résultats ne valident ni ce questionnaire, ni un usage réel. Le test
final historique a été mesuré sur le SFT successif précédent, pas sur ce modèle.

Le questionnaire pose des questions déterministes sur les champs absents.
La mention d'une douleur ajoute une question sur son retentissement ; cette
règle lexicale ne traite pas les négations et n'est pas un protocole clinique.
Une réponse incohérente, tronquée ou indisponible conduit à une revue humaine
sans attribution de priorité. Une sortie correctement structurée peut rester fausse.

## Exécution et traçabilité

Je réserve ZeroGPU à la génération finale ; le formulaire s'exécute sur CPU.
Je conserve une trace par session avec les entrées, la sortie, la provenance
du modèle et la durée. Elle est téléchargeable en JSON et temporaire sur le
serveur : **je la télécharge pour la conserver**. Il n'existe ni historique
partagé, ni base patient, ni garantie d'archivage ou d'immutabilité.

L'hébergement gratuit impose des quotas et une file d'attente. Cette variante
Gradio/Transformers complète la démonstration FastAPI/Docker/vLLM du dépôt du
projet ; elle ne démontre pas le fonctionnement de vLLM.

## Provenance

- Socle : [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), Apache 2.0.
- Adaptation triage : cas issus de [MediQAl, ANR-MALADES](https://huggingface.co/datasets/ANR-MALADES/MediQAl) (CC BY 4.0) et
  [UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) (MIT),
  transformés et annotés dans le projet OCR14. Aucun corpus brut n'est inclus.
- Interface et scripts préparés avec l'assistance de Codex.

[Fonctionnement de ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).
