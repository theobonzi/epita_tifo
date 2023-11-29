# Traitement d'Images

## Introduction
Ce projet est une implémentation de divers filtres de traitement d'images en temps réel, réalisé dans le cadre du cours de Traitement d'images à EPITA.

## Projet
Ce projet intègre une interface graphique qui utilise la caméra pour appliquer les différents filtres en temps réel. L'interface affiche également l'histogramme de l'image en temps réel et permet de personnaliser l'ordre d'activation des filtres.

## Filtres
L'objectif principal de ce projet est d'implémenter et de tester les filtres suivants en temps réel :
- Filtre Gris (Gray filter)
- Filtre Binaire (Binary filter)
- Égalisation d'histogramme (Histogram equalization)
- Filtre Négatif (Negative filter)
- Filtre de Pixelisation (Pixelize filter)
- Filtre de Sobel
- Filtre de Laplace
- Filtre de Prewitt
- Filtre Flou (Blur filter)
- Filtre d'Érosion (Erode filter)
- Filtre de Dilatation (Dilate filter)

## Utilisation et Installation
### Prérequis
Pour exécuter ce projet, vous devez avoir Python installé sur votre machine.

### Installation des Dépendances
Les dépendances nécessaires sont listées dans `requirements.txt`. Pour les installer, exécutez :
```bash
pip install -r requirements.txt
```

### Lancement du Projet
Pour démarrer l'interface de traitement d'images, exécutez :
```bash
python src/tifo.py
```

