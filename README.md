# 🖼️ Traitement d'Images

## Introduction
Ce projet, réalisé dans le cadre du cours de Traitement d'images à EPITA, implémente une variété de filtres appliqués en temps réel sur le flux vidéo de la caméra. Il offre une interface utilisateur pour changer l'ordre des filtres cumulables, allant des filtres simples aux convolutions plus complexes.

## Fonctionnalités
- **Filtres Simples** : Niveaux de gris, Binarisation, Égalisation d'histogramme, Négatif, Pixelisation.
- **Filtres de Détection de Bords** : Sobel, Laplace, Prewitt
- **Morphologie** : Erosion et Dilatation, combinables pour réaliser la fermeture et l'ouverture.
- **Histogramme en Temps Réel** : Visualisation dynamique de l'histogramme de l'image.

## Contraintes du Temps Réel
- Utilisation de **OpenCL** pour la parallélisation en kernels.
- Processus **SIMD** avec les tableaux Numpy.

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

