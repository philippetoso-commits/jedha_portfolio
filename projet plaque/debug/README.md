# Dossier debug

Ce dossier regroupe les scripts de **test, debug et outils ponctuels** utilisés durant le développement.

**Ces fichiers ne font PAS partie du runtime de l'application** (ils sont exclus du déploiement Docker / Hugging Face Spaces via `.dockerignore` et les scripts `deploy.sh` / `update_space.sh`).

## Contenu

### Scripts de test Gradio / OpenCV

| Fichier | Rôle |
|---|---|
| `gradio_test.py` | Hello-world Gradio (validation de l'install) |
| `test_video_display.py` | Test d'affichage vidéo dans Gradio |
| `test_codecs.py` | Test des codecs OpenCV disponibles |
| `test_video_quick.py` | Test rapide du pipeline vidéo |
| `test_gradio_flow.py` | Test du flow complet Gradio |

### Tests fonctionnels

| Fichier | Rôle |
|---|---|
| `test_pipeline.py` | Test du pipeline ALPR complet sur images |
| `test_admin_db.py` | Test interactif des opérations BDD |
| `debug_video.py` | Debug du rendu vidéo |

### Scripts utilitaires (one-shot)

| Fichier | Rôle |
|---|---|
| `extract_plates_batch.py` | Extraction batch de plaques (a généré `plaques_extraites.csv`) |
| `extract_demo_images.py` | Extraction d'images de démo depuis le dataset UC3M-LP |
| `generate_fake_data.py` | Génération de faux résidents (a généré `plaques_avec_donnees.csv`) |

### Fichiers de log et tests vidéo

- `app.log`, `app_log.txt` : logs runtime archivés
- `debug_keys.txt` : tokens de debug (à ne pas committer)
- `test_*.mp4` / `*.webm` / `*.avi` : vidéos générées par les tests de codec

## Note pour le jury

Le code de production se trouve dans `../projetplaquetransfert/` (entry point `app.py` + modules `utils/`).
