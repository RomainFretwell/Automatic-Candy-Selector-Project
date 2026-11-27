# Détection de bonbons (YOLO)

# à faire
- [x] faire dataset pour detection
- [x] séparer test et entrainement 20-80
- [x] choisir modèle et moteur pour la partie detection (YOLO11s pré-entraîné COCO, Ultralytics)
- [ ] entraîner modèle
- [ ] évaluer sur le jeu de test
- [ ] faire partie détection (script de prédiction)

# Dataset detection :
- sur plusieurs fonds random (blanc, bois, gris, bleu foncé, nappe, moquette...)
- faire varier le nombre de bonbon (3 à 11 par photo)
- rajouter decors et objets random
- 124 photos annotées avec Label Studio, 741 bonbons, 22 classes (`type_couleur`, mêmes noms que la classification)
- format YOLO : une image dans `images/` + un fichier texte du même nom dans `labels/`, une ligne par bonbon : `classe x_centre y_centre largeur hauteur` (valeurs entre 0 et 1)
- l'export Label Studio d'origine (4080x3060) est dans `export_label_studio/` (pas dans git, trop lourd) ;
  `src/importDataset.py` le convertit en images 1280 px dans `dataset/entrainement/` (99 photos) et `dataset/test/` (25 photos)
- les noms des classes sont dans `dataset/data.yaml`

# Utilisation (détection)
Depuis `Automatic-Candy-Selector-Project/detection/`, dans PowerShell :
```powershell
..\.venv\Scripts\Activate.ps1
python src\importDataset.py                  # importe export_label_studio\ (80 % entraînement, 20 % test)
python src\train.py                          # entraîne YOLO11s -> models\candyYolo11s.pt
python src\evaluate.py                       # précision, rappel, mAP sur le jeu de test
python src\predict.py dataset\test\images    # détecte les bonbons, images annotées dans runs\predict\
```
- le modèle `.pt` n'est pas dans git (trop lourd) : lancer `train.py` après un clone / pull
- `--help` sur chaque script pour voir les options
