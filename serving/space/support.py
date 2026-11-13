"""Présentation autonome du projet : HTML local et métriques agrégées, sans appel GPU."""

from html import escape
from pathlib import Path


def document():
    """Insère les résultats vérifiés dans la page, sans charger de ressource externe."""
    dossier = Path(__file__).with_suffix('')
    donnees = (dossier / 'resultats.json').read_text(encoding='utf-8').replace('<', '\\u003c')
    page = (dossier / 'index.html').read_text(encoding='utf-8')
    for marqueur, fichier in [('__STYLE__', 'style.css'), ('__SCRIPT__', 'script.js')]:
        page = page.replace(marqueur, (dossier / fichier).read_text(encoding='utf-8'))
    return page.replace('__RESULTATS__', donnees)


def iframe():
    """Isole le JavaScript pédagogique de la session et des données de la démo."""
    return ('<iframe title="Fonctionnement du projet OCR14" style="width:100%;height:920px;border:0;border-radius:12px" '
            'sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" '
            f'srcdoc="{escape(document(), quote=True)}"></iframe>')
