"""
Importe l'export YOLO de Label Studio (export_label_studio/) dans dataset/entrainement et dataset/test.

Pour chaque photo :
  - applique la rotation EXIF (Label Studio a annoté l'image tournée, comme on la voit à l'écran)
    puis enregistre sans EXIF -> tous les outils voient la même image que les boîtes ;
  - réduit le grand côté à 1280 px (YOLO s'entraîne en 640 px, inutile de garder du 4080x3060) ;
  - copie le fichier de boîtes (coordonnées relatives 0-1, inchangées par un redimensionnement).
Répartition 80-20 : 20 % des photos de chaque séance (même heure de prise de vue = même fond :
blanc, bois, nappe, moquette...) vont dans le test. Parmi 2000 tirages (graine fixe), on garde
celui où chaque classe de bonbon a le plus près de 20 % de ses exemplaires dans le test.
(un simple "une photo sur 5" ne marche pas : les couleurs de crocodile tournent sur 5 photos,
le test n'avait presque que des crocodiles orange)
Les noms de classes sont alignés sur ceux de la classification et écrits dans dataset/data.yaml.
Les photos déjà importées sont ignorées.

Usage (depuis detection/) :
    python src/importDataset.py
    python src/importDataset.py --dry-run
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

MAX_SIZE = 1280
# noms Label Studio -> noms de la classification (dossiers de classification/photo_bonbon)
RENAME = {
    "ourson_orangeclair": "ourson_orangeClair",
    "ourson_orangefonce": "ourson_orangeFonce",
    "schtroumpf_blanc": "schtroumpf_bleu-blanc",
    "schtroumpf_jaune": "schtroumpf_bleu-jaune",
    "schtroumpf_rouge": "schtroumpf_bleu-rouge",
}


def loadClassNames(source):
    # l'ordre de classes.txt donne le numéro de classe utilisé dans les fichiers de boîtes
    names = (source / "classes.txt").read_text(encoding="utf-8").split()
    return [RENAME.get(name, name) for name in names]


def photoDate(photo):
    # "00676181-20261125_151853.jpg" -> "20261125_151853" (le préfixe est ajouté par Label Studio)
    return photo.stem.split("-", 1)[-1]


def photoSession(photo):
    # "20261126_144317" -> "20261126_14" : photos prises la même heure = même séance = même fond
    return photoDate(photo)[:11]


def drawTest(sessions, testRatio, rng):
    # tire testRatio des photos de chaque séance
    return [p for s in sorted(sessions) for p in rng.sample(sessions[s], round(len(sessions[s]) * testRatio))]


def balanceError(test, classesOf, totals, testRatio):
    # écart entre la part de chaque classe dans le test et la part visée (0 = parfait)
    inTest = Counter(c for p in test for c in classesOf[p])
    return sum((inTest[c] / n - testRatio) ** 2 for c, n in totals.items())


def splitPhotos(photos, classesOf, testRatio, seed, tries=2000):
    # parmi `tries` tirages par séance, garde celui où chaque classe est le plus proche de testRatio
    rng = random.Random(seed)
    sessions = defaultdict(list)
    for p in sorted(photos):
        sessions[photoSession(p)].append(p)
    totals = Counter(c for p in photos for c in classesOf[p])
    candidates = (drawTest(sessions, testRatio, rng) for _ in range(tries))
    test = set(min(candidates, key=lambda t: balanceError(t, classesOf, totals, testRatio)))
    return [p for p in sorted(photos) if p not in test], sorted(test)


def readClasses(source, photo):
    # liste des numéros de classe des bonbons d'une photo (une ligne = un bonbon)
    lines = (source / "labels" / f"{photo.stem}.txt").read_text(encoding="utf-8").split("\n")
    return [int(line.split()[0]) for line in lines if line.strip()]


def checkLabels(labelFile, nClasses):
    # vérifie que chaque ligne est "classe x y w h" avec des valeurs valides
    lines = labelFile.read_text(encoding="utf-8").split("\n")
    for line in filter(None, (line.strip() for line in lines)):
        cls, *box = line.split()
        if len(box) != 4 or not 0 <= int(cls) < nClasses or not all(0 <= float(v) <= 1 for v in box):
            raise ValueError(f"Ligne invalide dans {labelFile.name} : {line}")


def resizePhoto(photo):
    img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
    return img


def importPhoto(photo, source, splitDir, nClasses):
    labelFile = source / "labels" / f"{photo.stem}.txt"
    checkLabels(labelFile, nClasses)
    name = photoDate(photo)
    resizePhoto(photo).save(splitDir / "images" / f"{name}.jpg", quality=90)
    (splitDir / "labels" / f"{name}.txt").write_text(labelFile.read_text(encoding="utf-8"), encoding="utf-8")


def writeDataYaml(datasetDir, classNames):
    # fichier de configuration pour Ultralytics (chemins relatifs au dossier de ce fichier ;
    # pas de chemin absolu pour qu'il marche sur tous les PC)
    lines = ["# généré par src/importDataset.py", "train: entrainement/images", "test: test/images", "names:"]
    lines += [f"    {i}: {name}" for i, name in enumerate(classNames)]
    (datasetDir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def importSplit(photos, source, splitDir, nClasses, dryRun):
    # importe les photos pas encore présentes dans splitDir ; renvoie le nombre de nouvelles
    for sub in ("images", "labels"):
        (splitDir / sub).mkdir(parents=True, exist_ok=True)
    new = [p for p in photos if not (splitDir / "images" / f"{photoDate(p)}.jpg").exists()]
    for photo in new if not dryRun else []:
        importPhoto(photo, source, splitDir, nClasses)
    return len(new)


def importAll(args):
    source, datasetDir = Path(args.source), Path(args.dataset)
    classNames = loadClassNames(source)
    photos = sorted((source / "images").glob("*.jpg"))
    classesOf = {p: readClasses(source, p) for p in photos}
    train, test = splitPhotos(photos, classesOf, args.test_ratio, args.seed)
    for splitName, splitPhotosList in (("entrainement", train), ("test", test)):
        n = importSplit(splitPhotosList, source, datasetDir / splitName, len(classNames), args.dry_run)
        print(f"{splitName} : {len(splitPhotosList)} photos dont {n} nouvelles")
    if not args.dry_run:
        writeDataYaml(datasetDir, classNames)
    print(f"{len(classNames)} classes" + (" (simulation, rien n'a été écrit)" if args.dry_run else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="export_label_studio", help="export YOLO de Label Studio")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0, help="graine du tirage entraînement / test")
    parser.add_argument("--dry-run", action="store_true", help="affiche la répartition sans rien écrire")
    importAll(parser.parse_args())


if __name__ == "__main__":
    main()
