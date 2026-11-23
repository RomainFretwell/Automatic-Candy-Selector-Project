# à faire
- [x] choisir un modèle (resnet50 ?)
- [x] choisir moteur d'exécution (pythorch ?)
- [x] tester sur des images qu'il connait (faire inférer)
- [x] faire un git
- [x] prendre des photos
- [x] trier photo dans dossiers par classe
- [x] séparer test et entrainement 20-80
- [x] passer les étiquettes en JSON (type + couleur dominante)
- [x] plus de photo en faisant des combinaisons de zoom et de rotation des photos qu'on a déjà
- [x] entraîner modèle
- [x] prendre toutes les photos pour la classification
- [ ] gérer les images qui ne sont pas des bonbons
- [ ] exporter le modèle (ONNX ?) pour l'appli Java
- [ ] faire l'appli java
- [ ] choisir modèle et moteur pour la partie detection
- [ ] faire dataset pour detection
- [ ] faire partie détection

# Choix du modèle
- ResNet-50
### avantages :
- moins de paramètres que VGG
- moins lourd que VGG
- un peu plus précis que VGG
- à peu près les mêmes stats que inception-v3 mais plus simple d'utilisation
- finetuning beaucoup plus facile qu'avec inception-v3
- ResNet très documenté car plus couramment utilisé (beaucoup de tuto)
- On peut utiliser "cuda" pour utiliser le gpu de l'ordinateur plutôt que le cpu

# Choix du moteur d'exécution
- pytorch
### avantages :
- plus simple à utiliser
- beaucoup de tutoriels

# Dataset classification :
- sur fond blanc avec plusieurs angles de photo
- 20 photos par bonbon + 100 variantes
- les variantes générées (rotation, zoom, luminosité) sont dans `dataset/entrainement/images_supplémentaires/`

# Dataset detection :
- sur plusieurs fonds random
- faire varier le nombre de bonbon
- rajouter decors et objets random

# Utilisation (classification)
Depuis `Automatic-Candy-Selector-Project/`, dans PowerShell :
```powershell
.\.venv\Scripts\Activate.ps1
python src\augmentDataset.py                 # crée les variantes manquantes
python src\train.py                          # entraîne le modèle -> models\candyResnet50.pt
python src\evaluate.py                       # précision sur le jeu de test + matrices de confusion
python src\predict.py dataset\test\images    # prédit type + couleur (image ou dossier)
```
- le modèle `.pt` n'est pas dans git (trop lourd) : lancer `train.py` après un clone / pull
- `--help` sur chaque script pour voir les options
