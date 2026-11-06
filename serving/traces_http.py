"""Corrélation des appels HTTP, sans corps de requête, clé ni URL libre."""

from contextvars import ContextVar

# FastAPI recopie le contexte dans son pool de threads : le client vLLM retrouve
# l'identifiant créé par le middleware, sans variable globale partagée par visiteur.
contexte_requete = ContextVar('ocr14_requete', default=None)


def tracer_sortie(route, statut, duree_ms, erreur_type=None):
    """Ajoute l'appel vLLM au même audit que la requête entrante."""
    contexte = contexte_requete.get()
    if contexte is not None:
        audit, identifiant = contexte
        audit.http(identifiant, 'vllm', route, statut, duree_ms, erreur_type)
