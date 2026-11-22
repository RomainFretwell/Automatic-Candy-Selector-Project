"""
Test simple de ResNet pré-entraîné sur ImageNet (PyTorch).
Usage :
    python test_resnet.py image1.jpg image2.png      -> une ou plusieurs images
    python test_resnet.py dossier_images/            -> toutes les images d'un dossier
    python test_resnet.py dossier_images/ --model resnet18
"""
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def charger_modele(nom, device):
    if nom == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        weights = ResNet50_Weights.IMAGENET1K_V1 # ou mettre DEFAULT pour avoir la version IMAGENET1K_V2
        model = resnet50(weights=weights)
    model.eval().to(device)
    # weights.transforms() = redimensionnement + recadrage + normalisation ImageNet attendus par le modèle
    return model, weights.transforms(), weights.meta["categories"]


def lister_images(args):
    images = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            images += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.is_file():
            images.append(p)
        else:
            print(f"[!] Introuvable : {a}")
    return images


def main():
    args = sys.argv[1:]
    nom_modele = "resnet50"
    if "--model" in args:
        i = args.index("--model")
        nom_modele = args[i + 1]
        args = args[:i] + args[i + 2:]

    if not args:
        print(__doc__)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} | device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model, preprocess, classes = charger_modele(nom_modele, device)
    print(f"Modèle : {nom_modele} (poids ImageNet, {len(classes)} classes)\n")

    for chemin in lister_images(args):
        img = Image.open(chemin).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)  # forme [1, 3, 224, 224]

        t0 = time.perf_counter()
        with torch.no_grad():
            probas = torch.softmax(model(x), dim=1)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        top5 = torch.topk(probas, 5)
        print(f"=== {chemin.name}  ({dt:.0f} ms)")
        for p, idx in zip(top5.values, top5.indices):
            print(f"   {p.item() * 100:5.1f} %  {classes[idx]}")
        print()


if __name__ == "__main__":
    main()
