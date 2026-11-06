"""Journal SQLite des interactions fictives du POC, distinct des logs techniques."""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


class Audit:
    """Conserve le début et le résultat d'une interaction, même si l'inférence échoue."""
    def __init__(self, chemin):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with self.connecter() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, date_utc TEXT NOT NULL,
                demande TEXT NOT NULL, resultat TEXT, modele TEXT NOT NULL, statut TEXT NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS evenements_http (
                id INTEGER PRIMARY KEY, request_id TEXT NOT NULL, date_utc TEXT NOT NULL,
                couche TEXT NOT NULL, route TEXT NOT NULL, statut INTEGER,
                duree_ms REAL, erreur_type TEXT)''')
        os.chmod(self.chemin, 0o600)

    def connecter(self):
        """Ouvre une connexion courte par opération ; SQLite sérialise les écritures."""
        return sqlite3.connect(self.chemin, timeout=10)

    def commencer(self, session_id, demande, modele):
        identifiant = str(uuid4())
        with self.connecter() as db:
            db.execute('INSERT INTO interactions VALUES (?, ?, ?, ?, NULL, ?, ?)',
                       (identifiant, session_id, datetime.now(timezone.utc).isoformat(),
                        json.dumps(demande, ensure_ascii=False), json.dumps(modele), 'en_cours'))
        return identifiant

    def http(self, request_id, couche, route, statut=None, duree_ms=None, erreur_type=None):
        """Journal technique minimal ; les détails fictifs restent dans l'audit métier."""
        evenement = dict(request_id=request_id, date_utc=datetime.now(timezone.utc).isoformat(),
                         couche=couche, route=route, statut=statut,
                         duree_ms=duree_ms, erreur_type=erreur_type)
        with self.connecter() as db:
            db.execute('''INSERT INTO evenements_http
                (request_id,date_utc,couche,route,statut,duree_ms,erreur_type)
                VALUES (:request_id,:date_utc,:couche,:route,:statut,:duree_ms,:erreur_type)''', evenement)
        logging.getLogger('uvicorn.error').info(json.dumps(evenement, ensure_ascii=False))

    def terminer(self, identifiant, resultat):
        with self.connecter() as db:
            db.execute('UPDATE interactions SET resultat = ?, statut = ? WHERE id = ?',
                       (json.dumps(resultat, ensure_ascii=False), resultat['statut'], identifiant))

    def lire(self, identifiant):
        with self.connecter() as db:
            db.row_factory = sqlite3.Row
            ligne = db.execute('SELECT * FROM interactions WHERE id = ?', (identifiant,)).fetchone()
        if ligne is None:
            return None
        resultat = dict(ligne)
        for champ in ('demande', 'resultat', 'modele'):
            resultat[champ] = json.loads(resultat[champ]) if resultat[champ] else None
        return resultat
