"""
Fine-tuning de ResNet-50 (2 sorties : type + couleur) sur dataset/entrainement.

Une partie des photos d'entraînement sert de validation pour garder le meilleur modèle
(le jeu de test n'est jamais utilisé ici, il sert seulement à evaluate.py).

Usage :
    python src/train.py
    python src/train.py --epochs 20 --lr 0.0001
Sorties (dossier models/) : candyResnet50.pt, trainingHistory.json, trainingCurves.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre seulement l'image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from candyDataset import CandyDataset, buildCombinations, buildVocab, splitTrainVal
from candyModel import CandyNet, getDevice, saveCheckpoint
from labelsIo import loadLabels


def parseArgs():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset/entrainement")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def makeLoaders(args):
    labels = loadLabels(args.data)
    typeClasses, colorClasses = buildVocab(labels)
    trainEntries, valEntries = splitTrainVal(labels, args.val_ratio, args.seed)
    trainSet = CandyDataset(args.data, trainEntries, typeClasses, colorClasses)
    valSet = CandyDataset(args.data, valEntries, typeClasses, colorClasses)
    print(f"Types : {typeClasses}\nCouleurs : {colorClasses}")
    combinations = buildCombinations(labels)
    print(f"Combinaisons valides : {len(combinations)}")
    print(f"Entraînement : {len(trainSet)} images | validation : {len(valSet)} photos d'origine\n")
    trainLoader = DataLoader(trainSet, batch_size=args.batch, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=args.batch)
    return trainLoader, valLoader, typeClasses, colorClasses, combinations


def runEpoch(model, loader, device, optimizer=None):
    # une passe sur le loader ; entraîne si un optimizer est donné, sinon évalue seulement
    training = optimizer is not None
    model.train(training)
    lossFn = nn.CrossEntropyLoss()
    totalLoss, typeOk, colorOk, n = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for x, yType, yColor in loader:
            x, yType, yColor = x.to(device), yType.to(device), yColor.to(device)
            typeLogits, colorLogits = model(x)
            # erreur totale = erreur sur le type + erreur sur la couleur
            loss = lossFn(typeLogits, yType) + lossFn(colorLogits, yColor)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            totalLoss += loss.item() * len(x)
            typeOk += (typeLogits.argmax(1) == yType).sum().item()
            colorOk += (colorLogits.argmax(1) == yColor).sum().item()
            n += len(x)
    return {"loss": totalLoss / n, "typeAcc": typeOk / n, "colorAcc": colorOk / n}


def plotHistory(history, outFile):
    epochs = range(1, len(history) + 1)
    fig, (axLoss, axAcc) = plt.subplots(1, 2, figsize=(11, 4))
    for split in ["train", "val"]:
        axLoss.plot(epochs, [h[split]["loss"] for h in history], label=split)
        axAcc.plot(epochs, [h[split]["typeAcc"] for h in history], label=f"type ({split})")
        axAcc.plot(epochs, [h[split]["colorAcc"] for h in history], "--", label=f"couleur ({split})")
    axLoss.set(title="Loss (type + couleur)", xlabel="époque")
    axAcc.set(title="Précision", xlabel="époque", ylim=(0, 1.05))
    axLoss.legend()
    axAcc.legend()
    fig.tight_layout()
    fig.savefig(outFile, dpi=120)


def isBetter(metrics, best):
    # meilleur = plus de bonnes réponses (type + couleur) en validation, puis loss plus faible
    score = (metrics["typeAcc"] + metrics["colorAcc"], -metrics["loss"])
    return best is None or score > best, score


def train(args):
    torch.manual_seed(args.seed)
    device = getDevice()
    outDir = Path(args.out)
    outDir.mkdir(exist_ok=True)
    trainLoader, valLoader, typeClasses, colorClasses, combinations = makeLoaders(args)
    model = CandyNet(len(typeClasses), len(colorClasses)).to(device)
    # fine-tuning complet avec un petit learning rate pour ne pas "casser" les poids ImageNet
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        trainMetrics = runEpoch(model, trainLoader, device, optimizer)
        valMetrics = runEpoch(model, valLoader, device)
        history.append({"epoch": epoch, "train": trainMetrics, "val": valMetrics})
        better, score = isBetter(valMetrics, best)
        if better:
            best = score
            saveCheckpoint(outDir / "candyResnet50.pt", model, typeClasses, colorClasses, combinations)
        print(f"Époque {epoch:2d} | train loss {trainMetrics['loss']:.3f} type {trainMetrics['typeAcc']:.0%}"
              f" | val loss {valMetrics['loss']:.3f} type {valMetrics['typeAcc']:.0%}"
              f" couleur {valMetrics['colorAcc']:.0%}" + ("  <- sauvegardé" if better else ""))
    (outDir / "trainingHistory.json").write_text(json.dumps(history, indent=4), encoding="utf-8")
    plotHistory(history, outDir / "trainingCurves.png")
    print(f"\nModèle : {outDir / 'candyResnet50.pt'} | courbes : {outDir / 'trainingCurves.png'}")


if __name__ == "__main__":
    train(parseArgs())
