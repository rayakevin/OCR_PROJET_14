"""Assemble une remise reproductible par liste fermée, sans secrets ni notes privées."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

RACINE=Path(__file__).resolve().parents[1]


def fichiers_code():
    """Inclut les scripts, tests et notebooks du projet ; exclut le diagnostic matériel."""
    extensions={'.py','.ipynb','.md','.json','.html','.txt','.yaml','.yml'}
    for dossier in ('data_ingestion','evaluation','training','serving','scripts','tests','.github'):
        for p in sorted((RACINE/dossier).rglob('*')):
            if (p.is_file() and '__pycache__' not in p.parts
                    and p.name!='test_monitoring.py'
                    and (p.suffix in extensions or p.name=='Dockerfile')):
                yield p
    for nom in ('README.md','pyproject.toml','uv.lock'):
        yield RACINE/nom


def conditionner(destination, paquet_hf):
    """Conserve les fichiers sources ; crée une nouvelle archive vérifiée à chaque export."""
    destination.mkdir(parents=True,exist_ok=False)
    nom='Triage_medical_bilingue_Raya_Kevin'
    dossier=destination/nom
    dossier.mkdir()
    for sous, source in [('Raya_Kevin_1_Dataset_102026',paquet_hf/'dataset'),
                         ('Raya_Kevin_2_Modeles_SFT_DPO_102026',paquet_hf/'modele')]:
        shutil.copytree(source,dossier/sous)
    shutil.copyfile(RACINE/'output/pdf/rapport_technique_ocr14.pdf',dossier/'Raya_Kevin_3_Rapport_technique_102026.pdf')
    code=dossier/'Raya_Kevin_4_Endpoint_CICD_102026'
    for p in fichiers_code():
        cible=code/p.relative_to(RACINE);cible.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,cible)
    oral=dossier/'Raya_Kevin_5_Support_oral_102026';oral.mkdir()
    for source,nom_sortie in [('output/support_soutenance.html','support_soutenance.html'),
                             ('reports/plan_soutenance.md','plan_soutenance.md')]:
        shutil.copyfile(RACINE/source,oral/nom_sortie)
    preuves=dossier/'preuves';preuves.mkdir()
    for nom_preuve in ('publication_livrables_hf.json','bilan_dpo_cible_20261106.json','bilan_final_20261105.json'):
        shutil.copyfile(RACINE/'reports'/nom_preuve,preuves/nom_preuve)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RACINE,text=True).strip()
    (dossier/'LIRE_AVANT_REMISE.md').write_text(f'''# Remise OCR14 — Kevin Raya

Démarrage retenu : octobre 2026 (premier commit : 16 octobre).
Révision Git de référence au conditionnement : {revision}.
Les empreintes du manifeste identifient les fichiers effectivement inclus.

1. Dataset : quatre configurations JSONL, partitions v3, provenance et empreintes.
2. Modèles : SFT mixte à la racine, SFT successif et DPO ciblés dans les sous-dossiers.
   Les DPO ciblés sont des essais non retenus ; je ne les présente pas comme meilleurs.
3. Rapport technique : PDF de 15 pages, inférieur à la limite de 20.
4. Code, notebooks et CI/CD : instructions dans le README ; données et poids à
   replacer selon les chemins documentés pour reconstruire l'environnement.
5. Support d'oral : page HTML autonome et plan de quinze minutes.

Démonstration : https://huggingface.co/spaces/rayakevin/ocr14-triage-fr-en
Dataset : https://huggingface.co/datasets/rayakevin/ocr14-medical-bilingue-v3
Modèles : https://huggingface.co/rayakevin/ocr14-triage-lora-experiences
Code / CI : https://github.com/rayakevin/OCR_PROJET_14

Les dépôts dataset, modèles et code sont privés : le lien seul n'accorde pas
l'accès. Le ZIP contient les fichiers de remise ; toute diffusion supplémentaire
des données demande de clarifier les licences non renseignées.

## Point restant avant de déclarer la remise intégralement conforme

Le Space public fonctionne avec Transformers/ZeroGPU. Le notebook 09 prépare
un endpoint vLLM HTTPS temporaire, protégé par clé, avec vérification et arrêt
automatiques. Sa vérification publique n'est pas encore acquise. Le test interne
sur Colab et la préparation du notebook ne remplacent pas cette preuve.

Aucune validation clinique n'a été menée. Les données de triage sont silver et
les cas bilingues sont corrélés. Le rapport détaille les limites et décisions.
Les notes personnelles, secrets, diagnostics matériels et données brutes sont exclus.
''')
    empreintes={str(p.relative_to(dossier)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(dossier.rglob('*')) if p.is_file()}
    (dossier/'MANIFESTE_SHA256.json').write_text(json.dumps(empreintes,ensure_ascii=False,indent=2)+'\n')
    archive=destination/(dossier.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dossier.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(destination))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Archive corrompue.')
        for chemin,sha in empreintes.items():
            if hashlib.sha256(z.read(dossier.name+'/'+chemin)).hexdigest()!=sha:
                raise RuntimeError('Empreinte ZIP différente : '+chemin)
    print(f'Archive vérifiée : {archive}\n{len(empreintes)} fichiers ; {archive.stat().st_size/1e6:.1f} Mo')
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=RACINE/'data/livrables/remise_20261106')
    p.add_argument('--paquet-hf',type=Path,default=RACINE/'data/deployment/livrables_hf_20261106_v3')
    a=p.parse_args();conditionner(a.destination,a.paquet_hf)
