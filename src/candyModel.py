"""
Modèle de classification des bonbons : ResNet-50 pré-entraîné (ImageNet) avec 2 sorties.
    - tête "type"    : Crocodile, Dragibus, ...
    - tête "couleur" : rouge, jaune, ...
Les deux têtes partagent le même extracteur de caractéristiques (le corps du ResNet),
on n'a donc qu'un seul modèle à entraîner et à exporter pour Java.
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


def saveCheckpoint(path, model, typeClasses, colorClasses):
    # on sauvegarde aussi les noms des classes : sans eux, les index prédits n'ont pas de sens
    torch.save({
        "state": model.state_dict(),
        "typeClasses": typeClasses,
        "colorClasses": colorClasses,
    }, path)


def loadCheckpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    typeClasses, colorClasses = checkpoint["typeClasses"], checkpoint["colorClasses"]
    model = CandyNet(len(typeClasses), len(colorClasses), pretrained=False)
    model.load_state_dict(checkpoint["state"])
    model.eval().to(device)
    return model, typeClasses, colorClasses


def getDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"
