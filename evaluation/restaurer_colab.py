"""Vérifie l'archive Colab et prépare une copie locale de son suivi MLflow."""

import hashlib
import json
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

RACINE = Path(__file__).resolve().parents[1]
ARCHIVE = RACINE / 'data/colab_archive'


def verifier_archive():
    """Compare chaque fichier inventorié à sa taille Drive et au hash du transfert."""
    inventaire = json.loads((ARCHIVE / 'inventory.json').read_text())
    transferts = {
        ligne['path']: ligne
        for texte in (ARCHIVE / 'transfers.jsonl').read_text().splitlines()
        if (ligne := json.loads(texte))['status'] == 'ok'
    }
    total = 0
    erreurs = []
    fichiers = [f for f in inventaire if f['file_or_folder'] == 'file']
    for fichier in fichiers:
        relatif = fichier['path']
        chemin = ARCHIVE / 'original' / relatif
        transfert = transferts.get(relatif)
        if not chemin.is_file() or transfert is None:
            erreurs.append(relatif)
            continue
        with chemin.open('rb') as flux:
            empreinte = hashlib.file_digest(flux, 'sha256').hexdigest()
        if (chemin.stat().st_size != int(fichier['size'])
                or empreinte != transfert['sha256']):
            erreurs.append(relatif)
        total += chemin.stat().st_size
    if erreurs:
        raise RuntimeError('Archive incomplète ou altérée : ' + ', '.join(erreurs))
    return {'fichiers': len(fichiers), 'octets': total}


def preparer_mlflow():
    """Réécrit seulement les URI d'artefacts dans une copie de la base originale.

    Les paramètres et métriques gardent leur provenance Colab. Les chemins
    historiques dans les paramètres restent donc inchangés ; seuls les liens
    utilisés par MLflow pour ouvrir les artefacts deviennent locaux.
    """
    original = ARCHIVE / 'original'
    destination = ARCHIVE / 'local/mlflow_colab.db'
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Une seconde exécution conserve les éventuels ajouts à la copie locale.
    if not destination.exists():
        with sqlite3.connect((original / 'suivi/mlflow_colab.db').as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(destination) as copie:
                source.backup(copie)
    ancien = 'file:///content/drive/MyDrive/OCR_PROJET_14'
    with sqlite3.connect(destination) as copie:
        if copie.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('La base SQLite est corrompue.')
        for table, colonne in [('experiments', 'artifact_location'), ('runs', 'artifact_uri')]:
            copie.execute(
                f'UPDATE {table} SET {colonne} = replace({colonne}, ?, ?) WHERE {colonne} LIKE ?',
                (ancien, original.as_uri(), ancien + '/%'),
            )
        # L'expérience Default est vide : aucun artefact Colab à récupérer ici.
        copie.execute("UPDATE experiments SET artifact_location = ? WHERE artifact_location = '/content/mlruns/0'",
                      ((ARCHIVE / 'local/mlruns/0').as_uri(),))
        etats = dict(copie.execute('SELECT status, COUNT(*) FROM runs GROUP BY status'))
        manquants = []
        for identifiant, statut, uri in copie.execute('SELECT run_uuid, status, artifact_uri FROM runs'):
            if statut == 'FINISHED' and not Path(unquote(urlparse(uri).path)).is_dir():
                manquants.append(identifiant)
    return {'base': str(destination.relative_to(RACINE)), 'executions': etats,
            'artefacts_absents_runs_termines': manquants}


def main():
    """Produit un constat vérifiable, sans téléchargement ni lancement GPU."""
    bilan = {**verifier_archive(), 'mlflow': preparer_mlflow()}
    limites = ARCHIVE / 'limites_recuperation.json'
    if limites.exists():
        bilan['limites_recuperation'] = json.loads(limites.read_text())
    (ARCHIVE / 'verification.json').write_text(
        json.dumps(bilan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(bilan, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
