"""
Évaluation du modèle entraîné sur le jeu de test (photos jamais vues pendant l'entraînement).

Usage :
    python src/evaluate.py
    python src/evaluate.py --model models/candyResnet50.pt --data dataset/test
Sorties (dossier models/) : testResults.json, confusionType.png, confusionColor.png
La réponse évaluée est la combinaison (type, couleur) valide la plus probable ; on la compare
aussi aux deux sorties prises séparément (ancienne méthode).
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
from candyModel import bestCombination, getDevice, loadCheckpoint


def predictAll(model, loader, device, classes):
    # vraies classes, réponse finale (combinaison valide) et sorties séparées ("raw"), en listes d'index
    p = {k: [] for k in ("trueType", "trueColor", "type", "color", "rawType", "rawColor")}
    with torch.no_grad():
        for x, yType, yColor in loader:
            typeLogits, colorLogits = model(x.to(device))
            typeIdx, colorIdx, _ = bestCombination(typeLogits, colorLogits, *classes)
            p["trueType"] += yType.tolist()
            p["trueColor"] += yColor.tolist()
            p["type"] += typeIdx.cpu().tolist()
            p["color"] += colorIdx.cpu().tolist()
            p["rawType"] += typeLogits.argmax(1).cpu().tolist()
            p["rawColor"] += colorLogits.argmax(1).cpu().tolist()
    return p


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


def listErrors(dataset, p, typeClasses, colorClasses):
    errors = []
    for i, entry in enumerate(dataset.entries):
        if p["trueType"][i] != p["type"][i] or p["trueColor"][i] != p["color"][i]:
            errors.append({"image": entry["image"],
                           "true": f"{typeClasses[p['trueType'][i]]} {colorClasses[p['trueColor'][i]]}",
                           "predicted": f"{typeClasses[p['type'][i]]} {colorClasses[p['color'][i]]}"})
    return errors


def accuracy(true, pred):
    return sum(t == p for t, p in zip(true, pred)) / len(true)


def accuracies(p, predType, predColor):
    # précision type, couleur et type ET couleur des prédictions données
    both = [tt == pt and tc == pc for tt, pt, tc, pc in zip(p["trueType"], predType, p["trueColor"], predColor)]
    return {"typeAccuracy": accuracy(p["trueType"], predType),
            "colorAccuracy": accuracy(p["trueColor"], predColor),
            "bothAccuracy": sum(both) / len(both)}


def countInvalidPairs(p, combinations, typeClasses, colorClasses):
    # nombre de paires (type, couleur) inexistantes données par les sorties séparées
    valid = {(t, c) for t, c in combinations}
    return sum((typeClasses[t], colorClasses[c]) not in valid for t, c in zip(p["rawType"], p["rawColor"]))


def printReport(results):
    print(f"Images de test : {results['nImages']}")
    print(f"Précision type    : {results['typeAccuracy']:.1%}")
    print(f"Précision couleur : {results['colorAccuracy']:.1%}")
    print(f"Type ET couleur   : {results['bothAccuracy']:.1%}")
    raw = results["independentHeads"]
    print(f"(sorties séparées : type {raw['typeAccuracy']:.1%}, couleur {raw['colorAccuracy']:.1%}, "
          f"les deux {raw['bothAccuracy']:.1%}, paires impossibles : {raw['invalidPairs']})\n")
    for title, key in (("Précision par type :", "typePerClass"), ("\nPrécision par couleur :", "colorPerClass")):
        print(title)
        for name, acc in results[key].items():
            print(f"   {name:12s} {acc:.0%}" if acc is not None else f"   {name:12s} (aucune image)")
    print(f"\nErreurs ({len(results['errors'])}) :")
    for e in results["errors"]:
        print(f"   {e['image']} : vrai = {e['true']}, prédit = {e['predicted']}")


def evaluate(args):
    device = getDevice()
    model, typeClasses, colorClasses, combinations = loadCheckpoint(args.model, device)
    dataset = loadSplit(args.data, typeClasses, colorClasses)
    classes = (combinations, typeClasses, colorClasses)
    p = predictAll(model, DataLoader(dataset, batch_size=32), device, classes)
    typeMatrix = confusionMatrix(p["trueType"], p["type"], len(typeClasses))
    colorMatrix = confusionMatrix(p["trueColor"], p["color"], len(colorClasses))
    independent = accuracies(p, p["rawType"], p["rawColor"])
    independent["invalidPairs"] = countInvalidPairs(p, combinations, typeClasses, colorClasses)
    results = {
        "nImages": len(dataset),
        "validCombinations": len(combinations),
        **accuracies(p, p["type"], p["color"]),
        "independentHeads": independent,
        "typePerClass": perClassAccuracy(typeMatrix, typeClasses),
        "colorPerClass": perClassAccuracy(colorMatrix, colorClasses),
        "typeConfusion": {"classes": typeClasses, "matrix": typeMatrix},
        "colorConfusion": {"classes": colorClasses, "matrix": colorMatrix},
        "errors": listErrors(dataset, p, typeClasses, colorClasses),
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
