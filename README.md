# à faire
- [x] choisir un modèle (resnet50 ?)
- [x] choisir moteur d'exécution (pythorch ?)
- [x] tester sur des images qu'il connait (faire inférer)
- [x] faire un git
- [x] prendre des photos
- [x] trier photo dans dossiers par classe
- [x] séparer test et entrainement 20-80
- [ ] plus de photo en faisant des combinaisons de zoom et de rotation des photos qu'on a déjà
- [ ] entraîner modèle

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
- mais il faudra quand même faire l'appli finale en java

# Dataset :
- sur fond blanc / fond random
- bcp lumière / pas bcp
- pas d'ombre / plusieurs angles d'ombre
- angles de photo
- cas le plus simple = photo prise du dessus + fond blanc + lumière du dessus (pas d'ombre et assez lumineux)
- [ ] à justifier : combien d'images ? combien par catégories ?
