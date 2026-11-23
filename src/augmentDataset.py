"""
Data augmentation "sur disque" du jeu d'entraînement.

Pour chaque image de <split>/images/, crée N variantes (rotation, zoom, luminosité)
dans <split>/images_supplémentaires/ et ajoute leurs étiquettes dans <split>/labels.json.
Le script est relançable : il ne génère que les variantes manquantes, et recopie
le type / la couleur de l'image d'origine (utile si on corrige une étiquette à la main).

Usage :
    python src/augmentDataset.py                         -> dataset/entrainement, 5 variantes
    python src/augmentDataset.py --n 8
    python src/augmentDataset.py --split dataset/entrainement
    python src/augmentDataset.py --force                 -> réécrit toutes les variantes
Ne jamais l'appliquer au jeu de test (il doit rester composé de vraies photos).
"""
import argparse
import zlib
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from labelsIo import loadLabels, saveLabels

AUG_DIR = "images_supplémentaires"
ROTATION_DEGREES = 180      # le bonbon est photographié du dessus : toutes les orientations sont réalistes
ZOOM_RANGE = (0.6, 1.3)     # bonbon plus petit (x0.6) ou plus gros (x1.3) que sur la photo importée
BRIGHTNESS = 0.2            # luminosité +/- 20 %


def makeVariant(img, seed):
    # graine fixe par variante -> mêmes images si on regénère (reproductible)
    torch.manual_seed(seed)
    w, h = img.size
    transform = v2.Compose([
        # on agrandit l'image en prolongeant ses bords : après rotation / dézoom, les coins
        # contiennent du "fond" réaliste au lieu de triangles noirs (pas de miroir :
        # en dézoomant, il ferait apparaître des copies du bonbon)
        v2.Pad(padding=w, padding_mode="edge"),
        v2.RandomAffine(degrees=ROTATION_DEGREES, scale=ZOOM_RANGE),
        v2.CenterCrop((h, w)),
        v2.ColorJitter(brightness=BRIGHTNESS),
    ])
    return transform(img)


def variantPath(original, k):
    return f"{AUG_DIR}/{Path(original).stem}_aug{k}.jpg"


def ensureVariant(splitDir, entry, k, force):
    # crée le fichier de la variante k s'il n'existe pas (ou le réécrit si force) ; renvoie True si écrit
    relPath = variantPath(entry["image"], k)
    outFile = splitDir / relPath
    if outFile.exists() and not force:
        return False
    img = Image.open(splitDir / entry["image"]).convert("RGB")
    seed = zlib.crc32(relPath.encode("utf-8"))
    makeVariant(img, seed).save(outFile, quality=95)
    return True


def variantEntry(entry, k):
    return {
        "image": variantPath(entry["image"], k),
        "type": entry["type"],
        "color": entry["color"],
        "augmentedFrom": entry["image"],
    }


def warnUnlabeled(splitDir, originals):
    # signale les photos ajoutées dans images/ mais pas encore étiquetées à la main
    labeled = {e["image"] for e in originals}
    for f in sorted((splitDir / "images").iterdir()):
        if f"images/{f.name}" not in labeled:
            print(f"[!] Image sans étiquette (ignorée) : images/{f.name}")


def warnOrphans(splitDir, labeledVariants):
    # variantes présentes sur le disque dont l'image d'origine n'est plus étiquetée
    for f in sorted((splitDir / AUG_DIR).iterdir()):
        if f"{AUG_DIR}/{f.name}" not in labeledVariants:
            print(f"[!] Variante orpheline (non supprimée) : {AUG_DIR}/{f.name}")


def augmentSplit(splitDir, n, force):
    labels = loadLabels(splitDir)
    originals = [e for e in labels if "augmentedFrom" not in e]
    (splitDir / AUG_DIR).mkdir(exist_ok=True)
    warnUnlabeled(splitDir, originals)

    variants, created = [], 0
    for entry in originals:
        for k in range(1, n + 1):
            created += ensureVariant(splitDir, entry, k, force)
            variants.append(variantEntry(entry, k))
    # les anciennes entrées de variantes sont remplacées par les nouvelles (étiquettes resynchronisées)
    saveLabels(splitDir, originals + variants)
    warnOrphans(splitDir, {v["image"] for v in variants})
    print(f"{len(originals)} images d'origine, {len(variants)} variantes ({created} créées)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dataset/entrainement", help="dossier contenant images/ et labels.json")
    parser.add_argument("--n", type=int, default=5, help="nombre de variantes par image")
    parser.add_argument("--force", action="store_true",
                        help="réécrit aussi les variantes existantes (si on change les paramètres)")
    args = parser.parse_args()
    augmentSplit(Path(args.split), args.n, args.force)


if __name__ == "__main__":
    main()
