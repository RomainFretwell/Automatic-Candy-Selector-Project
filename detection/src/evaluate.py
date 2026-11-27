"""
Évaluation du modèle de détection sur le jeu de test (photos jamais vues pendant l'entraînement).

Mesures (calculées par Ultralytics) :
  - précision : parmi les boîtes trouvées, part de boîtes correctes (bonne classe, bien placées) ;
  - rappel : parmi les bonbons de l'image, part de bonbons trouvés ;
  - mAP50 : qualité globale, une boîte compte comme bien placée si elle recouvre au moins 50 %
    de la vraie boîte (IoU >= 0.5) ; mAP50-95 : moyenne pour des seuils de 50 % à 95 % (plus exigeant).

Usage (depuis detection/) :
    python src/evaluate.py
    python src/evaluate.py --model models/candyYolo11s.pt --data dataset
Sorties (dossier models/) : testResults.json, confusionTest.png
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

def makeTestYaml(dataDir, runDir):
    # data.yaml dont la partie "val" est le jeu de test (Ultralytics évalue toujours sur "val")
    names = YAML.load(dataDir / "data.yaml")["names"]
    testImages = (dataDir / "test" / "images").resolve().as_posix()
    config = {"train": testImages, "val": testImages, "names": names}
    runDir.mkdir(parents=True, exist_ok=True)
    YAML.save(runDir / "data.yaml", config)
    return runDir / "data.yaml"


def perClass(box, names):
    # précision, rappel et mAP50 de chaque classe présente dans le test
    result = {}
    for i, cls in enumerate(box.ap_class_index):
        result[names[int(cls)]] = {"precision": float(box.p[i]), "recall": float(box.r[i]),
                                   "mAP50": float(box.ap50[i])}
    return result


def summarize(metrics, names):
    box = metrics.box
    return {"precision": float(box.mp), "recall": float(box.mr), "mAP50": float(box.map50),
            "mAP50-95": float(box.map), "perClass": perClass(box, names)}


def printReport(results):
    print(f"\nPrécision : {results['precision']:.1%} | rappel : {results['recall']:.1%}"
          f" | mAP50 : {results['mAP50']:.1%} | mAP50-95 : {results['mAP50-95']:.1%}\n")
    print(f"   {'classe':24s} précision  rappel  mAP50")
    for name, m in results["perClass"].items():
        print(f"   {name:24s} {m['precision']:8.0%} {m['recall']:7.0%} {m['mAP50']:6.0%}")


def evaluate(args):
    runDir = Path("runs") / "test"
    dataYaml = makeTestYaml(Path(args.data), runDir)
    model = YOLO(args.model)
    metrics = model.val(data=str(dataYaml), imgsz=args.imgsz, batch=8, split="val",
                        project=str(runDir.resolve()), name="val", exist_ok=True)
    results = summarize(metrics, model.names)
    printReport(results)
    outDir = Path(args.out)
    (outDir / "testResults.json").write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
    shutil.copy(Path(metrics.save_dir) / "confusion_matrix.png", outDir / "confusionTest.png")
    print(f"\nRésultats : {outDir / 'testResults.json'}, {outDir / 'confusionTest.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/candyYolo11s.pt")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="models")
    parser.add_argument("--imgsz", type=int, default=640, help="même valeur qu'à l'entraînement")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
