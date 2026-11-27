"""
Détection des bonbons sur une ou plusieurs images : liste chaque bonbon trouvé (classe, confiance,
position) et enregistre une copie de l'image avec les boîtes dessinées.

Usage (depuis detection/) :
    python src/predict.py image.jpg
    python src/predict.py dossier_images/
    python src/predict.py image.jpg --conf 0.5 --out predictions
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def listImages(paths):
    images = []
    for p in map(Path, paths):
        if p.is_dir():
            images += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.is_file():
            images.append(p)
        else:
            print(f"[!] Introuvable : {p}")
    return images


def loadImage(imagePath):
    # rotation EXIF appliquée (photo de téléphone) : même orientation que les images d'entraînement
    return ImageOps.exif_transpose(Image.open(imagePath)).convert("RGB")


def printDetections(imagePath, result):
    print(f"=== {imagePath.name} : {len(result.boxes)} bonbon(s)")
    for cls, conf, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        x1, y1, x2, y2 = (int(v) for v in box)
        print(f"   {result.names[int(cls)]:24s} {conf:.0%}   ({x1}, {y1}) -> ({x2}, {y2})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="images ou dossiers d'images")
    parser.add_argument("--model", default="models/candyYolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confiance minimale d'une boîte")
    parser.add_argument("--imgsz", type=int, default=640, help="même valeur qu'à l'entraînement")
    parser.add_argument("--out", default="runs/predict", help="dossier des images annotées")
    args = parser.parse_args()
    model = YOLO(args.model)
    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    for imagePath in listImages(args.images):
        result = model.predict(loadImage(imagePath), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        printDetections(imagePath, result)
        result.save(filename=str(outDir / f"{imagePath.stem}.jpg"))
    print(f"\nImages annotées : {outDir}")


if __name__ == "__main__":
    main()
