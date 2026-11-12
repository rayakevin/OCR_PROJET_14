"""Interface traduite selon la langue choisie, sans appel GPU lors des réglages."""

import tempfile
from pathlib import Path

import gradio as gr
from serving.domaine import Consultation, prochaine_question
from serving.space.parcours import avancer, exporter_trace, nouvelle_session
from serving.space.support import iframe

TEXTES = {
    'fr': {
        'titre': '# Démonstration de triage',
        'intro': '**Cas fictifs uniquement.** Prototype pédagogique : toute proposition doit être vérifiée par un professionnel.',
        'demarche': 'Je collecte les informations, puis le modèle propose un niveau ESI et une priorité. Les quotas ZeroGPU peuvent limiter les essais.',
        'reglages': 'Réglages', 'mode': 'Mode de saisie',
        'modes': [('Questionnaire guidé', 'guide'), ('Cas complet', 'complet')],
        'details': 'Afficher les détails techniques',
        'aide': 'Le mode « Cas complet » analyse directement la présentation saisie. Changer de langue ou de mode commence un nouveau cas.',
        'question': 'Question', 'reponse': 'Votre réponse', 'envoyer': 'Continuer',
        'nouveau': 'Nouveau cas', 'resultat': 'Proposition', 'technique': 'Détails techniques',
        'trace': 'Télécharger la trace JSON', 'fin': 'Proposition à vérifier',
        'cas_complet': 'Décrivez le cas complet : symptômes, début, évolution et contexte connu. Indiquez les informations inconnues.',
        'analyser': 'Analyser le cas',
        'conservation': '**Conservation :** la trace est temporaire. Je la télécharge avant de quitter la page pour la conserver. Aucun historique partagé entre visiteurs.',
        'invalide': 'Vérifiez la saisie : réponse non vide, longueur autorisée et absence de séparateurs de modèle.',
        'indisponible': 'Inférence indisponible ; aucune priorité attribuée.',
        'incoherent': 'Sortie incomplète, ambiguë ou incohérente.',
    },
    'en': {
        'titre': '# Triage demonstration',
        'intro': '**Fictional cases only.** Educational prototype: every proposal requires professional review.',
        'demarche': 'I collect information, then the model proposes an ESI level and a priority. ZeroGPU quotas may limit the number of attempts.',
        'reglages': 'Settings', 'mode': 'Input mode',
        'modes': [('Guided questionnaire', 'guide'), ('Complete case', 'complet')],
        'details': 'Show technical details',
        'aide': '“Complete case” analyses the entered presentation directly. Changing the language or input mode starts a new case.',
        'question': 'Question', 'reponse': 'Your answer', 'envoyer': 'Continue',
        'nouveau': 'New case', 'resultat': 'Proposal', 'technique': 'Technical details',
        'trace': 'Download JSON trace', 'fin': 'Proposal requiring review',
        'cas_complet': 'Describe the complete case: symptoms, onset, progression and known background. State what is unknown.',
        'analyser': 'Analyse case',
        'conservation': '**Retention:** the trace is temporary. I download it before leaving the page to keep a copy. No shared history between visitors.',
        'invalide': 'Check your input: a non-empty answer, within the length limit, without model delimiters.',
        'indisponible': 'Inference unavailable; no priority assigned.',
        'incoherent': 'Incomplete, ambiguous or inconsistent output.',
    },
}


def creer_interface(generer):
    """Traduit les textes visibles ; la trace conserve son schéma technique stable."""
    def demarrer(langue, mode):
        t = TEXTES[langue]
        etat = nouvelle_session(langue)
        q = (t['cas_complet'] if mode == 'complet' else
             prochaine_question(Consultation.model_validate(etat['cas']))['texte'])
        return (etat, gr.update(value=q, label=t['question']),
                gr.update(value='', label=t['reponse']), gr.update(value='', label=t['resultat']),
                gr.update(value=None, label=t['technique']), gr.update(value=None, label=t['trace']),
                gr.update(value=t['analyser'] if mode == 'complet' else t['envoyer'], interactive=True))

    def traduire(langue, mode):
        t = TEXTES[langue]
        return (*demarrer(langue, mode), t['titre'], t['intro'], t['demarche'],
                gr.update(label=t['reglages']), gr.update(choices=t['modes'], label=t['mode'], value=mode),
                gr.update(label=t['details']), t['aide'], gr.update(value=t['nouveau']), t['conservation'])

    def soumettre(texte, etat, mode):
        t = TEXTES[etat['cas']['langue']]
        try:
            suivant = avancer(texte, etat, generer, mode=mode)
        except ValueError:
            raise gr.Error(t['invalide']) from None
        reponse = suivant['evenements'][-1]['reponse']
        q = reponse.get('question', {}).get('texte', t['fin'])
        proposition = reponse.get('texte_modele', '')
        if reponse['statut'] == 'revue_humaine':
            message = t['indisponible'] if 'erreur_type' in reponse else t['incoherent']
            proposition = message + ('\n\n' + proposition if proposition else '')
        Path(trace.GRADIO_CACHE).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='trace-', dir=trace.GRADIO_CACHE,
                                         encoding='utf-8', delete=False) as fichier:
            fichier.write(exporter_trace(suivant))
        return (suivant, q, '', proposition, reponse, fichier.name,
                gr.update(interactive=not suivant['termine']))

    t = TEXTES['fr']
    with gr.Blocks(title='OCR14 · Triage', analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        with gr.Tab('Démonstration / Demo', id='demo'):
            with gr.Row():
                with gr.Column(scale=3):
                    titre = gr.Markdown(t['titre'])
                with gr.Column(scale=1, min_width=200):
                    langue = gr.Radio([('Français', 'fr'), ('English', 'en')], value='fr',
                                      label='Language', show_label=False, container=False)
            intro = gr.Markdown(t['intro'])
            demarche = gr.Markdown(t['demarche'])
            with gr.Accordion(t['reglages'], open=False) as reglages:
                mode = gr.Radio(t['modes'], value='guide', label=t['mode'])
                details = gr.Checkbox(value=False, label=t['details'])
                aide = gr.Markdown(t['aide'])
            etat = gr.State(lambda: nouvelle_session('fr'), time_to_live=3600)
            question = gr.Textbox(value=prochaine_question(Consultation())['texte'],
                                  label=t['question'], interactive=False)
            texte = gr.Textbox(label=t['reponse'], lines=4, max_length=6000)
            with gr.Row():
                envoyer = gr.Button(t['envoyer'], variant='primary')
                nouveau = gr.Button(t['nouveau'])
            proposition = gr.Textbox(label=t['resultat'], lines=5, interactive=False)
            resultat = gr.JSON(label=t['technique'], open=True, visible=False)
            trace = gr.File(label=t['trace'], interactive=False)
            conservation = gr.Markdown(t['conservation'])
            sorties = [etat, question, texte, proposition, resultat, trace, envoyer]
            envoi = envoyer.click(soumettre, [texte, etat, mode], sorties, concurrency_limit=1, api_name=False)
            nouveau.click(demarrer, [langue, mode], sorties, cancels=[envoi], api_name=False)
            mode.input(demarrer, [langue, mode], sorties, cancels=[envoi], api_name=False)
            langue.input(traduire, [langue, mode],
                         sorties + [titre, intro, demarche, reglages, mode, details, aide, nouveau, conservation],
                         cancels=[envoi], api_name=False)
            details.change(lambda visible: gr.update(visible=visible), details, resultat, api_name=False)
        with gr.Tab('Comprendre le projet (FR)', id='soutenance'):
            gr.HTML(iframe())
    return demo.queue(max_size=16)
