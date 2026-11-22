"""
Lecture / écriture des fichiers labels.json du dataset.

Format : liste d'objets
    {"image": "images/xxx.jpg", "type": "Crocodile", "color": "rouge"}
Le chemin "image" est relatif au dossier qui contient labels.json.
Les images générées par augmentDataset.py ont en plus "augmentedFrom" (chemin de l'image d'origine).
"""
import json
from pathlib import Path

LABELS_FILE = "labels.json"


def loadLabels(splitDir):
    path = Path(splitDir) / LABELS_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def saveLabels(splitDir, labels):
    # tri par chemin pour avoir un fichier stable (diffs git lisibles)
    labels = sorted(labels, key=lambda e: e["image"])
    text = json.dumps(labels, indent=4, ensure_ascii=False) + "\n"
    (Path(splitDir) / LABELS_FILE).write_text(text, encoding="utf-8")
