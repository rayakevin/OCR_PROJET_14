"""J'évalue les candidats figés sans modifier les poids ni sélectionner sur le test."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mlflow

RACINE = Path(__file__).resolve().parents[1]
PROTOCOLE = Path(__file__).with_name('protocole_test_final.json')
DPO_SHA256 = 'ae3cb4d55f3df9df573b031ccc8842fae9bbbddf2c634bd2e95fe2486d51cecc'


def empreinte(chemin):
    """Identifie le contenu exact, indépendamment de son emplacement."""
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--dpo-adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracking-uri', required=True)
    parser.add_argument('--artifact-location', required=True)
    args = parser.parse_args()
    protocole = json.loads(PROTOCOLE.read_text())
    for nom, attendu in protocole['partitions'].items():
        fichier = args.data_dir / f'{nom}.jsonl'
        assert empreinte(fichier) == attendu['sha256'], f'Partition différente : {nom}'
        assert len(fichier.read_text().splitlines()) == attendu['nombre']
    assert empreinte(args.adapter / 'adapter_model.safetensors') == protocole['sft_sha256']
    modeles = [('socle', None), ('sft_triage_v3', args.adapter)]
    if args.dpo_adapter:
        assert empreinte(args.dpo_adapter / 'adapter_model.safetensors') == DPO_SHA256
        modeles.append(('dpo_pilote_v3', args.dpo_adapter))
    args.output.mkdir(parents=True, exist_ok=True)
    copie = args.output / PROTOCOLE.name
    if copie.exists():
        assert copie.read_bytes() == PROTOCOLE.read_bytes(), 'Protocole différent.'
    else:
        copie.write_bytes(PROTOCOLE.read_bytes())
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experience = client.get_experiment_by_name('OCR14-Test-final-v3')
    experience_id = experience.experiment_id if experience else client.create_experiment(
        'OCR14-Test-final-v3', artifact_location=args.artifact_location)
    bilan = {'protocole': protocole, 'modeles': {}}
    for nom, adaptateur in modeles:
        dossier = args.output / nom
        dossier.mkdir(exist_ok=True)
        bilan['modeles'][nom] = {}
        for phase, tokens in (('qa', 256), ('triage', 192)):
            predictions = dossier / f'{phase}_test.jsonl'
            commande = [sys.executable, str(RACINE / 'evaluation/inference.py'),
                '--phase', phase, '--split', 'test', '--resume', '--data-dir', str(args.data_dir),
                '--revision', protocole['revision'], '--dtype', 'bfloat16', '--batch-size', '32',
                '--max-new-tokens', str(tokens), '--stop-token', 'eos', '--output', str(predictions)]
            if adaptateur:
                commande += ['--adapter', str(adaptateur)]
            with mlflow.start_run(experiment_id=experience_id, run_name=f'{nom}-{phase}-test'):
                mlflow.log_params({'modele': nom, 'phase': phase, 'split': 'test',
                    'revision': protocole['revision'], 'batch_size': 32, 'dtype': 'bfloat16',
                    'max_new_tokens': tokens, 'stop_token': 'eos', 'protocole_sha256': empreinte(PROTOCOLE),
                    'adapter_sha256': empreinte(adaptateur / 'adapter_model.safetensors') if adaptateur else 'aucun'})
                print(f'Évaluation finale : {nom} / {phase}', flush=True)
                journal = dossier / f'{phase}_execution.log'
                with journal.open('a', encoding='utf-8') as log:
                    subprocess.run(commande, cwd=RACINE, check=True, stdout=log, stderr=subprocess.STDOUT)
                    subprocess.run([sys.executable, str(RACINE / 'evaluation/evaluer.py'), str(predictions),
                        '--phase', phase, '--split', 'test', '--data-dir', str(args.data_dir)],
                        cwd=RACINE, check=True, stdout=log, stderr=subprocess.STDOUT)
                prefixe = 'phase1_qa' if phase == 'qa' else 'phase2_triage'
                references = [json.loads(l) for l in (args.data_dir / f'{prefixe}_test.jsonl').read_text().splitlines()]
                sorties = [json.loads(l) for l in predictions.read_text().splitlines()]
                assert [x['id'] for x in sorties] == [x['id'] for x in references]
                rapport = predictions.with_name(predictions.stem + '_metriques.json')
                valeurs = json.loads(rapport.read_text())
                scores = ({'accuracy_qcm': valeurs['qcm']['accuracy'],
                    'f1_lexical': valeurs['reponses_ouvertes']['f1_lexical_moyen']} if phase == 'qa' else {
                    'macro_f1_priorite': valeurs['priorite']['macro_f1'],
                    'accuracy_esi': valeurs['esi']['accuracy'],
                    'rappel_urgence_maximale': valeurs['priorite']['rappel_par_classe']['maximale']})
                scores['taux_eos'] = sum(x['arret'] == 'eos' for x in sorties) / len(sorties)
                mlflow.log_metrics(scores)
                for fichier in (predictions, rapport, journal, copie):
                    mlflow.log_artifact(str(fichier))
                bilan['modeles'][nom][phase] = valeurs
                print(nom, phase, scores, flush=True)
        # Un bilan intermédiaire conserve les modèles entièrement terminés.
        (args.output / 'bilan_test.json').write_text(json.dumps(bilan, ensure_ascii=False, indent=2) + '\n')
    print('Test final terminé. Aucun réentraînement ni changement de candidat.', flush=True)


if __name__ == '__main__':
    main()
