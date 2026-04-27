# Reconnaissance de Plaques d'Immatriculation (ALPR) 🚗📸

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![YOLO](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=yolo&logoColor=000)](#)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF4F8B?style=flat&logo=pytorch&logoColor=fff)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=000)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Ce projet met en œuvre un pipeline complet de détection et de reconnaissance de plaques d'immatriculation (ALPR - Automatic License Plate Recognition) en utilisant la Computer Vision et le Deep Learning.

> ⚠️ **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp.

---

## 📖 Le projet en quelques mots

L'objectif est de développer une solution en deux étapes :
1. **Détection (LP Detection)** : Localiser la position exacte de la plaque d'immatriculation dans une image à l'aide d'un réseau de neurones.
2. **Reconnaissance (OCR)** : Extraire et lire les caractères textuels présents sur la zone détectée.

---

## 📊 Les sources de données et RGPD

Les données utilisées pour l'entraînement proviennent du dataset académique public **UC3M-LP** de l'Université Carlos III de Madrid (hébergé sur Roboflow, Licence CC BY 4.0).

**⚖️ RGPD et Éthique :** 
Ce projet s’inscrit strictement en tant que réutilisation de données publiques à des fins pédagogiques. Le traitement est limité à la détection de plaques, sans identification de personnes, conformément aux principes de minimisation du RGPD.

---

## ⚙️ Installation

```bash
cd "projet plaque"
pip install ultralytics opencv-python fast-plate-ocr pandas
```

*Note : Pour le déploiement de l'application via Hugging Face, référez-vous au fichier `README_LIVRABLE.md` situé dans le dossier `projetplaquetransfert/`.*

---

## 🚀 L'Architecture et la Modélisation

Le projet ne part pas de zéro (from scratch). Il repose sur la technique du **Transfer Learning** (Apprentissage par Transfert).

### 1. Modèle de Détection (YOLOv8)
- **Modèle de base** : `yolov8n.pt` (pré-entraîné sur COCO).
- **Fine-Tuning** : L'architecture a été gelée partiellement (319/355 couches transférées) pour l'adapter spécifiquement à la détection de plaques sur 100 époques.
- **Résultat** : Un modèle robuste capable de détecter des plaques dans des conditions variées (jour/nuit, angles).

### 2. Reconnaissance de Caractères (OCR)
- Une fois la plaque détectée par YOLO, l'image "rognée" est envoyée à la librairie spécialisée `fast-plate-ocr` (un modèle global entraîné sur plus de 220 000 plaques) pour extraire le texte avec précision.

---

## 📂 Structure du projet

```text
projet plaque/
├── ALPR_YOLOv8_FastPlateOCR.ipynb     # Notebook complet (Détection + OCR)
├── Connect_roboflow.ipynb             # Script de récupération des données
├── EDA_Complete.ipynb                 # Exploration des images et bounding boxes
├── projetplaquetransfert/             # Dossier contenant le code prêt au déploiement (HuggingFace)
├── README_LIVRABLE.md                 # Guide pour déployer le dossier ci-dessus
└── README.md                          # Ce fichier
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
