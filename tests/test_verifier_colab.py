"""Je vérifie le démarrage HTTP avec 8080 occupé, sans charger de modèle."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from serving import verifier_colab


@unittest.skipUnless(os.name == 'posix', 'Colab transmet une socket POSIX à Uvicorn.')
class PortColabTests(unittest.TestCase):
    def test_api_demarre_quand_8080_est_occupe(self):
        # Si 8080 est déjà utilisé, le conflit existe sans créer de second serveur.
        occupation = socket.socket()
        self.addCleanup(occupation.close)
        try:
            occupation.bind(('127.0.0.1', 8080))
            occupation.listen()
        except OSError:
            occupation.close()

        lancer = subprocess.Popen
        attendre = verifier_colab.attendre

        def processus(commande, **options):
            if 'serving.lancer_vllm' in commande:
                commande = [sys.executable, '-c', 'import time; time.sleep(60)']
            return lancer(commande, **options)

        def disponibilite(url, cle, processus, delai):
            if url.endswith('/v1/models'):
                return  # Le modèle est remplacé ; seule l'écoute HTTP est testée.
            return attendre(url.replace('/readyz', '/healthz'), cle, processus, delai)

        with tempfile.TemporaryDirectory() as dossier:
            with patch.object(sys, 'argv', ['verifier', '--adapter', dossier, '--output', dossier]), \
                    patch.object(verifier_colab.subprocess, 'Popen', side_effect=processus), \
                    patch.object(verifier_colab, 'attendre', side_effect=disponibilite), \
                    patch.object(verifier_colab, 'verifier', return_value={}) as verifier:
                verifier_colab.main()
            self.assertNotEqual(verifier.call_args.args[0], 'http://127.0.0.1:8080')
            resultat = json.loads((Path(dossier) / 'verification_endpoint.json').read_text())
            self.assertEqual(resultat['controles_http'], {
                'sans_cle': 401, 'entree_invalide': 422, 'cas_incomplet': 'a_completer'})


if __name__ == '__main__':
    unittest.main()
