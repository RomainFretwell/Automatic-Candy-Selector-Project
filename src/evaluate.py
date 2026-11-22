"""
Évaluation du modèle entraîné sur le jeu de test (photos jamais vues pendant l'entraînement).

Usage :
    python src/evaluate.py
    python src/evaluate.py --model models/candyResnet50.pt --data dataset/test
Sorties (dossier models/) : testResults.json, confusionType.png, confusionColor.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from candyDataset import loadSplit
from candyModel import getDevice, loadCheckpoint


def predictAll(model, loader, device):
    # renvoie les vraies classes et les classes prédites, pour le type et la couleur
    trueType, predType, trueColor, predColor = [], [], [], []
    with torch.no_grad():
        for x, yType, yColor in loader:
            typeLogits, colorLogits = model(x.to(device))
            trueType += yType.tolist()
            predType += typeLogits.argmax(1).cpu().tolist()
            trueColor += yColor.tolist()
            predColor += colorLogits.argmax(1).cpu().tolist()
    return trueType, predType, trueColor, predColor


def confusionMatrix(true, pred, nClasses):
    # matrix[vraie classe][classe prédite] = nombre d'images
    matrix = [[0] * nClasses for _ in range(nClasses)]
    for t, p in zip(true, pred):
        matrix[t][p] += 1
    return matrix


def perClassAccuracy(matrix, classes):
    return {name: (row[i] / sum(row) if sum(row) else None) for i, (name, row) in enumerate(zip(classes, matrix))}


def plotConfusion(matrix, classes, title, outFile):
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 1.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="prédit", ylabel="vrai", title=title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center", color="white" if value > max(map(max, matrix)) / 2 else "black")
    fig.tight_layout()
    fig.savefig(outFile, dpi=120)


def listErrors(dataset, trueType, predType, trueColor, predColor, typeClasses, colorClasses):
    errors = []
    for i, entry in enumerate(dataset.entries):
        if trueType[i] != predType[i] or trueColor[i] != predColor[i]:
            errors.append({"image": entry["image"],
                           "true": f"{typeClasses[trueType[i]]} {colorClasses[trueColor[i]]}",
                           "predicted": f"{typeClasses[predType[i]]} {colorClasses[predColor[i]]}"})
    return errors


def accuracy(true, pred):
    return sum(t == p for t, p in zip(true, pred)) / len(true)


def printReport(results):
    print(f"Images de test : {results['nImages']}")
    print(f"Précision type    : {results['typeAccuracy']:.1%}")
    print(f"Précision couleur : {results['colorAccuracy']:.1%}")
    print(f"Type ET couleur   : {results['bothAccuracy']:.1%}\n")
    print("Précision par type :")
    for name, acc in results["typePerClass"].items():
        print(f"   {name:12s} {acc:.0%}" if acc is not None else f"   {name:12s} (aucune image)")
    print(f"\nErreurs ({len(results['errors'])}) :")
    for e in results["errors"]:
        print(f"   {e['image']} : vrai = {e['true']}, prédit = {e['predicted']}")


def evaluate(args):
    device = getDevice()
    model, typeClasses, colorClasses = loadCheckpoint(args.model, device)
    dataset = loadSplit(args.data, typeClasses, colorClasses)
    trueType, predType, trueColor, predColor = predictAll(model, DataLoader(dataset, batch_size=32), device)
    typeMatrix = confusionMatrix(trueType, predType, len(typeClasses))
    colorMatrix = confusionMatrix(trueColor, predColor, len(colorClasses))
    both = [tt == pt and tc == pc for tt, pt, tc, pc in zip(trueType, predType, trueColor, predColor)]
    results = {
        "nImages": len(dataset),
        "typeAccuracy": accuracy(trueType, predType),
        "colorAccuracy": accuracy(trueColor, predColor),
        "bothAccuracy": sum(both) / len(both),
        "typePerClass": perClassAccuracy(typeMatrix, typeClasses),
        "colorPerClass": perClassAccuracy(colorMatrix, colorClasses),
        "typeConfusion": {"classes": typeClasses, "matrix": typeMatrix},
        "colorConfusion": {"classes": colorClasses, "matrix": colorMatrix},
        "errors": listErrors(dataset, trueType, predType, trueColor, predColor, typeClasses, colorClasses),
    }
    return results, typeMatrix, colorMatrix, typeClasses, colorClasses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/candyResnet50.pt")
    parser.add_argument("--data", default="dataset/test")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    results, typeMatrix, colorMatrix, typeClasses, colorClasses = evaluate(args)
    printReport(results)
    outDir = Path(args.out)
    (outDir / "testResults.json").write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
    plotConfusion(typeMatrix, typeClasses, "Matrice de confusion - type", outDir / "confusionType.png")
    plotConfusion(colorMatrix, colorClasses, "Matrice de confusion - couleur", outDir / "confusionColor.png")
    print(f"\nRésultats : {outDir / 'testResults.json'}, {outDir / 'confusionType.png'}, {outDir / 'confusionColor.png'}")


if __name__ == "__main__":
    main()
