"""Exporte uniquement des métriques agrégées pour le support public de soutenance."""

import hashlib
import json
from pathlib import Path
import sqlite3

from evaluation.comparer_dpo_triage import scores

RACINE = Path(__file__).resolve().parents[1]
ARCHIVE = RACINE / 'data/colab_archive'


def construire():
    """Relie chaque ligne aux rapports archivés ; aucun cas ni prompt n'est publié."""
    lignes = []
    def ajouter(nom, groupe, dossier, epoque=None):
        fichiers = [dossier / f'{phase}_validation_metriques.json' for phase in ('qa', 'triage')]
        valeurs = scores(*(json.loads(p.read_text()) for p in fichiers))
        lignes.append({'nom': nom, 'groupe': groupe, 'epoque': epoque, 'scores': valeurs,
                       'sources': {str(p.relative_to(RACINE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in fichiers}})
    historique = ARCHIVE / 'original/experiences/sft_dpo_v3_20261105/comparaison'
    ajouter('SFT successif v3', 'historique', historique / 'sft_triage')
    ajouter('DPO généraliste', 'historique', RACINE / 'data/evaluation/colab_sft_dpo_v3/comparaison/dpo_pilote')
    for epoch in (1, 2, 3):
        ajouter(f'Mixte · époque {epoch}', 'mixte', ARCHIVE / 'sft_mixte_20261106_091447/validation' / f'checkpoint-{250*epoch}', epoch)
    for epoch in range(4, 11):
        ajouter(f'Poursuite · époque {epoch}', 'poursuite', ARCHIVE / 'poursuite_mixte_20261106_095443/validation' / f'checkpoint-{250*(epoch-3)}', epoch)
    for nom, dossier in [('Référence mixte réévaluée', 'sft_mixte_epoque3'), ('DPO ciblé · 16 pas', 'dpo_pas_16'), ('DPO ciblé · 32 pas', 'dpo_pas_32')]:
        ajouter(nom, 'dpo_cible', ARCHIVE / 'dpo_triage_cible_20261106_132004/validation' / dossier)
    # Courbes brutes de loss, avec leur step : pas de lissage ni de sélection favorable.
    courbes = []
    for nom in ('sft_mixte_20261106_091447', 'poursuite_mixte_20261106_095443', 'dpo_triage_cible_20261106_132004'):
        with sqlite3.connect((ARCHIVE / nom / 'mlflow.db').as_uri() + '?mode=ro', uri=True) as db:
            for run_id, key in db.execute("SELECT DISTINCT run_uuid,key FROM metrics WHERE key IN ('train/loss','eval/loss','loss','eval_loss')"):
                points = db.execute('SELECT step,value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step,timestamp', (run_id, key)).fetchall()
                courbes.append({'experience': nom, 'run_id': run_id, 'metrique': key, 'points': points})
    return {'date': '2026-11-06', 'split': 'validation v3', 'qa': 262, 'triage': 238,
            'limite': 'Références silver ; 119 paires de triage FR/EN corrélées. Aucun test clinique.',
            'lignes': lignes, 'courbes_mlflow': courbes}


if __name__ == '__main__':
    sortie = RACINE / 'serving/space/support/resultats.json'
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(construire(), ensure_ascii=False, indent=2) + '\n')
    print(sortie)
