"""
Chargement du dataset de bonbons (labels.json) pour PyTorch.
Chaque élément renvoyé : (image normalisée [3, 224, 224], index du type, index de la couleur).
"""
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from labelsIo import loadLabels

IMAGE_SIZE = 224
# moyenne / écart-type d'ImageNet : obligatoires car ResNet-50 a été pré-entraîné avec
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imageTransform():
    # prétraitement volontairement simple (redimension + normalisation) pour être
    # facile à reproduire plus tard dans l'application Java
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def buildVocab(labels):
    # listes triées des types et couleurs présents -> l'index dans la liste = numéro de classe
    types = sorted({e["type"] for e in labels})
    colors = sorted({e["color"] for e in labels})
    return types, colors


def splitTrainVal(labels, valRatio, seed=0):
    """
    Met de côté une partie des photos d'origine pour la validation (stratifié par type).
    Les variantes augmentées d'une photo de validation sont exclues de l'entraînement,
    sinon le modèle aurait déjà "vu" la photo et la validation serait trop optimiste.
    """
    rng = random.Random(seed)
    originalsByType = defaultdict(list)
    for e in labels:
        if "augmentedFrom" not in e:
            originalsByType[e["type"]].append(e["image"])
    valImages = set()
    for images in originalsByType.values():
        nVal = max(1, round(len(images) * valRatio))
        valImages.update(rng.sample(sorted(images), nVal))
    val = [e for e in labels if e["image"] in valImages]
    train = [e for e in labels if e["image"] not in valImages and e.get("augmentedFrom") not in valImages]
    return train, val


class CandyDataset(Dataset):
    def __init__(self, splitDir, entries, typeClasses, colorClasses):
        self.splitDir = Path(splitDir)
        self.entries = entries
        self.typeIndex = {name: i for i, name in enumerate(typeClasses)}
        self.colorIndex = {name: i for i, name in enumerate(colorClasses)}
        self.transform = imageTransform()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(self.splitDir / e["image"]).convert("RGB")
        return self.transform(img), self.typeIndex[e["type"]], self.colorIndex[e["color"]]


def loadSplit(splitDir, typeClasses, colorClasses):
    # charge tout un split (ex : le jeu de test) en ignorant les étiquettes inconnues du modèle
    labels = loadLabels(splitDir)
    known = [e for e in labels if e["type"] in typeClasses and e["color"] in colorClasses]
    for e in labels:
        if e not in known:
            print(f"[!] Étiquette inconnue du modèle, image ignorée : {e['image']} ({e['type']}, {e['color']})")
    return CandyDataset(splitDir, known, typeClasses, colorClasses)
