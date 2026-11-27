"""
Fine-tuning de YOLO11s (pré-entraîné sur COCO) sur dataset/entrainement.

Une partie des photos d'entraînement sert de validation pour garder le meilleur modèle
(même méthode que la répartition entraînement / test : par séance photo, classes équilibrées).
Le jeu de test n'est jamais utilisé ici, il sert seulement à evaluate.py.

Usage (depuis detection/) :
    python src/train.py
    python src/train.py --epochs 150 --imgsz 960
Sorties : models/candyYolo11s.pt + courbes et matrice de confusion de validation dans models/,
          détail de l'entraînement dans runs/<nom>/ (pas dans git)
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

from importDataset import readClasses, splitPhotos

# fichiers d'Ultralytics recopiés dans models/ pour être versionnés
REPORT_FILES = ["results.csv", "results.png", "confusion_matrix.png", "BoxPR_curve.png"]


def parseArgs():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--model", default="models/yolo11s.pt", help="poids pré-entraînés (téléchargés si absents)")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="taille des images vues par le réseau")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="train", help="sous-dossier de runs/")
    return parser.parse_args()


def writeImageList(photos, listFile):
    listFile.write_text("\n".join(p.resolve().as_posix() for p in photos) + "\n", encoding="utf-8")


def makeDataYaml(args, runDir):
    # crée un data.yaml d'entraînement : listes d'images entraînement / validation + noms des classes
    dataDir = Path(args.data)
    names = YAML.load(dataDir / "data.yaml")["names"]
    trainDir = dataDir / "entrainement"
    photos = sorted((trainDir / "images").glob("*.jpg"))
    classesOf = {p: readClasses(trainDir, p) for p in photos}
    train, val = splitPhotos(photos, classesOf, args.val_ratio, args.seed)
    runDir.mkdir(parents=True, exist_ok=True)
    writeImageList(train, runDir / "train.txt")
    writeImageList(val, runDir / "val.txt")
    config = {"train": (runDir / "train.txt").resolve().as_posix(),
              "val": (runDir / "val.txt").resolve().as_posix(), "names": names}
    YAML.save(runDir / "data.yaml", config)
    print(f"Entraînement : {len(train)} photos | validation : {len(val)} photos | {len(names)} classes\n")
    return runDir / "data.yaml"


def copyResults(trainDir, outDir):
    # meilleur modèle (selon la validation) + graphiques utiles pour le rapport
    outDir.mkdir(exist_ok=True)
    shutil.copy(trainDir / "weights" / "best.pt", outDir / "candyYolo11s.pt")
    for name in REPORT_FILES:
        if (trainDir / name).exists():
            shutil.copy(trainDir / name, outDir / name)


def train(args):
    runDir = Path("runs") / args.name
    dataYaml = makeDataYaml(args, runDir)
    Path(args.model).parent.mkdir(exist_ok=True)
    model = YOLO(args.model)
    # augmentation intégrée d'Ultralytics (mosaïque, échelle, couleurs, miroir...) laissée par défaut
    result = model.train(data=str(dataYaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                         seed=args.seed, project=str(runDir.resolve()), name="yolo", exist_ok=True)
    copyResults(Path(result.save_dir), Path(args.out))
    print(f"\nModèle : {Path(args.out) / 'candyYolo11s.pt'} | détails : {result.save_dir}")


if __name__ == "__main__":
    train(parseArgs())
