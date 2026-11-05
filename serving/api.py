"""API du POC : collecte progressive, proposition de triage et traçabilité."""

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import sqlite3
import time
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from serving.audit import Audit
from serving.backend import EntreeTropLongue, MODELE, Vllm
from serving.domaine import Consultation, interpreter, presentation, prochaine_question


AVERTISSEMENT = 'Prototype pédagogique — cas fictifs uniquement. Toute proposition doit être vérifiée par un soignant.'


def creer_app(*, backend=None, cle=None, audit_path=None):
    """Injecte les dépendances pour tester sans GPU ; aucun faux modèle en production."""
    secret = cle if cle is not None else os.environ.get('OCR14_API_KEY', '')
    stockage = audit_path or os.environ.get('OCR14_AUDIT_PATH', 'data/audit/interactions.sqlite3')

    @asynccontextmanager
    async def duree(app):
        if len(secret) < 24:
            raise RuntimeError('Définir OCR14_API_KEY avec au moins 24 caractères.')
        app.state.audit = Audit(stockage)
        if backend is None:
            if not os.environ.get('VLLM_API_KEY'):
                raise RuntimeError('Définir VLLM_API_KEY pour le serveur d’inférence.')
            app.state.backend = Vllm(os.environ.get('VLLM_URL', 'http://127.0.0.1:8001/v1'),
                                     os.environ['VLLM_API_KEY'])
        else:
            app.state.backend = backend
        yield
        if backend is None:
            app.state.backend.fermer()

    app = FastAPI(title='OCR14 — POC de triage', version='0.1.0', lifespan=duree)
    bearer = HTTPBearer(auto_error=False)

    def authentifier(jeton: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if jeton is None or not hmac.compare_digest(jeton.credentials.encode(), secret.encode()):
            raise HTTPException(401, 'Authentification requise.', headers={'WWW-Authenticate': 'Bearer'})

    @app.exception_handler(sqlite3.Error)
    async def erreur_audit(requete, erreur):
        # Si la trace ne peut pas être persistée, je ne délivre pas de décision.
        return JSONResponse(status_code=503, content={'detail': 'Journal indisponible ; traitement suspendu.'})

    @app.get('/', include_in_schema=False)
    def interface():
        return FileResponse(Path(__file__).with_name('interface.html'))

    @app.get('/healthz')
    def sante():
        return {'statut': 'api_active', 'inference_verifiee': False}

    @app.get('/readyz', dependencies=[Depends(authentifier)])
    def disponibilite():
        try:
            pret = app.state.backend.disponible()
        except Exception:
            pret = False
        if not pret:
            raise HTTPException(503, 'Le modèle attendu n’est pas disponible.')
        return {'statut': 'pret', 'modele': MODELE['id']}

    @app.post('/v1/consultations', dependencies=[Depends(authentifier)])
    def consulter(cas: Consultation):
        debut = time.perf_counter()
        session = str(cas.session_id or uuid4())
        audit = app.state.audit
        provenance = {k: MODELE[k] for k in ('id', 'revision', 'adapter_sha256', 'format_sft', 'mlflow_run_id')}
        identifiant = audit.commencer(session, cas.model_dump(mode='json'), provenance)
        resultat = {'interaction_id': identifiant, 'session_id': session, 'modele': provenance,
                    'avertissement': AVERTISSEMENT, 'validation_humaine_requise': True}
        question = prochaine_question(cas)
        code = 200
        if question:
            resultat.update(statut='a_completer', question=question, esi=None, priorite=None)
        else:
            try:
                generation = app.state.backend.generer(presentation(cas), cas.langue)
                resultat.update(interpreter(generation['texte'], generation['fin']))
                resultat.update(inference=generation)
            except EntreeTropLongue as erreur:
                code = 422
                resultat.update(statut='entree_trop_longue', motif=str(erreur), esi=None, priorite=None)
            except Exception as erreur:
                # Le détail HTTP du serveur peut contenir le prompt : je ne l'expose pas.
                code = 503
                resultat.update(statut='inference_indisponible', esi=None, priorite=None,
                                motif='Aucune proposition disponible ; évaluation humaine nécessaire.',
                                erreur_type=type(erreur).__name__)
        resultat['latence_ms'] = round(1000 * (time.perf_counter() - debut), 2)
        audit.terminer(identifiant, resultat)
        return JSONResponse(status_code=code, content=resultat)

    @app.get('/v1/interactions/{identifiant}', dependencies=[Depends(authentifier)])
    def interaction(identifiant: UUID):
        resultat = app.state.audit.lire(str(identifiant))
        if resultat is None:
            raise HTTPException(404, 'Interaction inconnue.')
        return resultat

    return app


app = creer_app()
