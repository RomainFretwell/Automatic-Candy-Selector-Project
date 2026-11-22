"""
Prédiction du type et de la couleur d'un bonbon sur une ou plusieurs images.

Usage :
    python src/predict.py image.jpg
    python src/predict.py dossier_images/
    python src/predict.py image.jpg --model models/candyResnet50.pt
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from candyDataset import imageTransform
from candyModel import getDevice, loadCheckpoint

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def listImages(paths):
    images = []
    for p in map(Path, paths):
        if p.is_dir():
            images += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.is_file():
            images.append(p)
        else:
            print(f"[!] Introuvable : {p}")
    return images


def predictImage(model, imagePath, device):
    # renvoie (index type, proba type, index couleur, proba couleur)
    x = imageTransform()(Image.open(imagePath).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        typeLogits, colorLogits = model(x)
    typeProbas = torch.softmax(typeLogits, dim=1)[0]
    colorProbas = torch.softmax(colorLogits, dim=1)[0]
    return typeProbas.argmax().item(), typeProbas.max().item(), colorProbas.argmax().item(), colorProbas.max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="images ou dossiers d'images")
    parser.add_argument("--model", default="models/candyResnet50.pt")
    args = parser.parse_args()
    device = getDevice()
    model, typeClasses, colorClasses = loadCheckpoint(args.model, device)
    for imagePath in listImages(args.images):
        typeIdx, typeConf, colorIdx, colorConf = predictImage(model, imagePath, device)
        print(f"=== {imagePath.name}")
        print(f"   Type : {typeClasses[typeIdx]}")
        print(f"   Couleur : {colorClasses[colorIdx]}")
        print(f"   Confiance : type {typeConf:.0%} / couleur {colorConf:.0%}\n")


if __name__ == "__main__":
    main()
