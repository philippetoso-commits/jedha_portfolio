#!/usr/bin/env python3
"""
Zip livrable jury (léger). Exclut jeux brut, caches, artefacts de tests,
et les gros checkpoints « from scratch » (~100 Mo) pour rester sous les plafonds
des outils de transfert (ex. 100 Mo). L'app reste fonctionnelle avec best.pt /
yolov8n.pt ; les poids hors zip sont recuperables depuis le depot Git/Git LFS ou HF.

Usage depuis n'importe quel repertoire :
    python3 "projet plaque/scripts/package_jury_zip.py"

Sortie par defaut : <parent de projet plaque>/projet_plaque_livrable_jury.zip
"""
import os
import sys
import time
import zipfile
from pathlib import Path
from fnmatch import fnmatch

# projet plaque/
ROOT = Path(__file__).resolve().parent.parent
# jedha_fullstack/
WORKSPACE_PARENT = ROOT.parent
OUTPUT = WORKSPACE_PARENT / "projet_plaque_livrable_jury.zip"

EXCLUDE_DIRS = {
    "venv",
    "__pycache__",
    ".git",
    "hf_space_deployment",
    "License-Plate-Recognition-4",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}

EXCLUDE_FILES_GLOB = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*:Zone.Identifier",
    "*:mshield",
    "*.DS_Store",
    "alpr.db",
    "License Plate Recognition.v4-resized640_aug3x-accurate.yolov8.zip",
    "projetplaquetransfert.tar.gz",
    "sample-20260211T133051Z-1-001.zip",
]

EXCLUDE_FILES_PATH_GLOB = [
    "*/outputs/*",
    "*/testvideo/*",
    "*/debug/test_*.mp4",
    "*/debug/test_*.webm",
    "*/debug/test_*.avi",
    "*/debug/test_output_video.mp4",
]

# Depasse le quota 100 Mo des transferts : disponibles via Git LFS / release / HF Space
SKIP_PLATE_WEIGHTS_IN_MODELS_DIR = {
    "modelemaison.pt",
    "YOLO_From_Scratch_LicensePlatev2.pt",
    "my_best_model.pt",  # doublon de modelcar/my_best_model.pt (chemins pipeline)
}


def should_exclude_heavy_models(rel_path: Path) -> bool:
    if rel_path.name not in SKIP_PLATE_WEIGHTS_IN_MODELS_DIR:
        return False
    parts = rel_path.parts
    if len(parts) < 3:
        return False
    # .../projetplaquetransfert/models/fichier.pt (pas sous-dossier genre models/foo/x.pt)
    return parts[-2] == "models" and parts[-3] == "projetplaquetransfert"


def should_exclude_file(rel_path: Path) -> bool:
    if should_exclude_heavy_models(rel_path):
        return True
    name = rel_path.name
    for pat in EXCLUDE_FILES_GLOB:
        if fnmatch(name, pat):
            return True
    full = "/" + str(rel_path).replace(os.sep, "/")
    for pat in EXCLUDE_FILES_PATH_GLOB:
        if fnmatch(full, pat):
            return True
    return False


def main() -> None:
    if not ROOT.exists():
        print(f"ERROR: dossier projet introuvable : {ROOT}", file=sys.stderr)
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    total_files = 0
    total_size = 0
    skipped = 0
    skipped_size = 0
    start = time.time()

    print(f"Creation du zip : {OUTPUT}")
    print(f"Source : {ROOT}\n")

    # compresslevel 9 : gain modeste sur les binaires, utile sur md/ipynb/py
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                try:
                    rel = fpath.relative_to(WORKSPACE_PARENT)
                except ValueError:
                    print(f"[skip hors workspace] {fpath}")
                    skipped += 1
                    continue
                if should_exclude_file(rel):
                    skipped += 1
                    try:
                        skipped_size += fpath.stat().st_size
                    except OSError:
                        pass
                    continue
                try:
                    sz = fpath.stat().st_size
                    zf.write(fpath, arcname=rel.as_posix())
                    total_files += 1
                    total_size += sz
                    if total_files % 200 == 0:
                        mb = total_size / 1024 / 1024
                        print(f"  ... {total_files} fichiers ({mb:.0f} Mo brut)")
                except (OSError, PermissionError) as exc:
                    print(f"  [skip] {rel} : {exc}")
                    skipped += 1

    elapsed = time.time() - start
    out_size = OUTPUT.stat().st_size
    out_mb = out_size / 1024 / 1024

    print("\n=== Resume ===")
    print(f"Fichiers inclus : {total_files}")
    print(f"Fichiers exclus : {skipped}")
    print(f"Taille brute incluse : {total_size / 1024 / 1024:.1f} Mo")
    print(f"Taille brute exclue  : {skipped_size / 1024 / 1024:.1f} Mo")
    print(f"Taille du zip final  : {out_mb:.1f} Mo")
    if total_size > 0:
        print(f"Taux compression brut : {(1 - out_size / total_size) * 100:.1f} %")
    print(f"Duree : {elapsed:.1f} s")
    print(f"\nSortie : {OUTPUT}")

    if out_mb > 100:
        print(
            "\nAttention : le zip depasse encore 100 Mo."
            " Exclure sample/, eda_outputs/ ou les notebooks multimodel si besoin.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
