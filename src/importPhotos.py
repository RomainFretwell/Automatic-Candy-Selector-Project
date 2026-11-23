"""
Importe les photos haute résolution de photo_bonbon/ dans le dataset (répartition 80-20).

photo_bonbon/ contient un sous-dossier par combinaison type + couleur, nommé type_couleur
(ex : crocodile_bleu, schtroumpf_bleu-blanc). Pour chaque photo pas encore importée :
    - remise dans le bon sens (rotation EXIF du téléphone) ;
    - carré central rogné puis réduit en 224x224 (le bonbon est à peu près centré) ;
    - enregistrée dans <split>/images/<dossier>_<nom>.jpg et ajoutée au labels.json du split.
Les originaux restent dans photo_bonbon/ (hors git) : on peut relancer le script après
chaque séance photo, seules les nouvelles photos sont importées.

Usage :
    python src/importPhotos.py --dry-run     # affiche ce qui serait importé
    python src/importPhotos.py
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageOps

from labelsIo import loadLabels, saveLabels

EXTS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 224


def parseFolderName(name):
    # "schtroumpf_bleu-blanc" -> ("Schtroumpf", "bleu-blanc") ; None si mal nommé
    # la couleur est gardée telle qu'écrite dans le nom du dossier (ex : orangeClair)
    candyType, sep, color = name.partition("_")
    if not sep or not candyType or not color:
        return None
    return candyType.capitalize(), color


def toSquare(img, size):
    # rotation EXIF puis carré central réduit en size x size
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)


def splitPhotos(photos, testRatio, seed):
    # tirage reproductible : ~testRatio des photos en test (au moins 1 s'il y en a 2 ou plus)
    photos = sorted(photos)
    random.Random(seed).shuffle(photos)
    nTest = round(len(photos) * testRatio)
    if nTest == 0 and len(photos) >= 2:
        nTest = 1
    return photos[nTest:], photos[:nTest]


def newPhotos(folder, imported):
    # photos du dossier dont la version réduite n'est pas encore dans un labels.json
    photos = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
    return [p for p in photos if targetName(folder, p) not in imported]


def targetName(folder, photo):
    return f"images/{folder.name}_{photo.stem}.jpg"


def importPhoto(photo, folder, splitDir, candyType, color):
    name = targetName(folder, photo)
    with Image.open(photo) as img:
        toSquare(img, IMAGE_SIZE).save(Path(splitDir) / name, quality=95)
    return {"image": name, "type": candyType, "color": color}


def importFolder(folder, splits, labels, args):
    # importe les nouvelles photos d'un sous-dossier, renvoie (nb entraînement, nb test)
    candyType, color = parseFolderName(folder.name)
    imported = {e["image"] for split in labels.values() for e in split}
    train, test = splitPhotos(newPhotos(folder, imported), args.test_ratio, f"{args.seed}-{folder.name}")
    if not args.dry_run:
        for photos, split in ((train, "train"), (test, "test")):
            for photo in photos:
                labels[split].append(importPhoto(photo, folder, splits[split], candyType, color))
    print(f"   {folder.name:24s} {candyType:11s} {color:11s} +{len(train)} entraînement, +{len(test)} test")
    return len(train), len(test)


def importAll(args):
    splits = {"train": args.train, "test": args.test}
    labels = {split: loadLabels(d) for split, d in splits.items()}
    folders = sorted(d for d in Path(args.source).iterdir() if d.is_dir())
    totals = [0, 0]
    for folder in folders:
        if parseFolderName(folder.name) is None:
            print(f"[!] Dossier ignoré (nom attendu : type_couleur) : {folder.name}")
            continue
        nTrain, nTest = importFolder(folder, splits, labels, args)
        totals[0] += nTrain
        totals[1] += nTest
    if not args.dry_run:
        for split, d in splits.items():
            saveLabels(d, labels[split])
    action = "Serait importé" if args.dry_run else "Importé"
    print(f"\n{action} : {totals[0]} en entraînement, {totals[1]} en test")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="photo_bonbon")
    parser.add_argument("--train", default="dataset/entrainement")
    parser.add_argument("--test", default="dataset/test")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", default="0", help="graine du tirage entraînement / test")
    parser.add_argument("--dry-run", action="store_true", help="n'écrit rien, affiche seulement")
    importAll(parser.parse_args())


if __name__ == "__main__":
    main()
