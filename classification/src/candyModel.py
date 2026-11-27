"""
Modèle de classification des bonbons : ResNet-50 pré-entraîné (ImageNet) avec 2 sorties.
    - tête "type"    : Crocodile, Dragibus, ...
    - tête "couleur" : rouge, jaune, ...
Les deux têtes partagent le même extracteur de caractéristiques (le corps du ResNet),
on n'a donc qu'un seul modèle à entraîner et à exporter pour Java.
À la prédiction, on ne répond qu'une combinaison (type, couleur) qui existe dans le dataset
(voir bestCombination) : jamais "Schtroumpf bleu" par exemple.
"""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class CandyNet(nn.Module):
    def __init__(self, nTypes, nColors, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        nFeatures = self.backbone.fc.in_features  # 2048 pour ResNet-50
        self.backbone.fc = nn.Identity()          # on retire la couche finale ImageNet (1000 classes)
        self.typeHead = nn.Linear(nFeatures, nTypes)
        self.colorHead = nn.Linear(nFeatures, nColors)

    def forward(self, x):
        features = self.backbone(x)
        return self.typeHead(features), self.colorHead(features)


def saveCheckpoint(path, model, typeClasses, colorClasses, combinations):
    # on sauvegarde aussi les noms des classes (sans eux, les index prédits n'ont pas de sens)
    # et les combinaisons (type, couleur) valides vues à l'entraînement
    torch.save({
        "state": model.state_dict(),
        "typeClasses": typeClasses,
        "colorClasses": colorClasses,
        "combinations": combinations,
    }, path)


def loadCheckpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    typeClasses, colorClasses = checkpoint["typeClasses"], checkpoint["colorClasses"]
    model = CandyNet(len(typeClasses), len(colorClasses), pretrained=False)
    model.load_state_dict(checkpoint["state"])
    model.eval().to(device)
    # ancien checkpoint sans combinaisons : toutes les paires sont permises (= comportement d'avant)
    combinations = checkpoint.get("combinations") or [[t, c] for t in typeClasses for c in colorClasses]
    return model, typeClasses, colorClasses, combinations


def bestCombination(typeLogits, colorLogits, combinations, typeClasses, colorClasses):
    """
    Choisit, pour chaque image du batch, la combinaison valide la plus probable.
    score(type, couleur) = P(type) x P(couleur), calculé seulement pour les combinaisons valides ;
    confiance = score de la meilleure / somme des scores des combinaisons valides.
    Renvoie (index type, index couleur, confiance), des tenseurs de taille [batch].
    """
    comboTypes = torch.tensor([typeClasses.index(t) for t, _ in combinations], device=typeLogits.device)
    comboColors = torch.tensor([colorClasses.index(c) for _, c in combinations], device=typeLogits.device)
    scores = typeLogits.softmax(1)[:, comboTypes] * colorLogits.softmax(1)[:, comboColors]
    confidence, best = (scores / scores.sum(1, keepdim=True)).max(1)
    return comboTypes[best], comboColors[best], confidence


def getDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"
