# Projet : Reconnaissance Automatique de Plaques d'Immatriculation (ALPR)

Ce projet met en œuvre un pipeline complet de détection et de reconnaissance de plaques d'immatriculation utilisant l'apprentissage profond (Deep Learning).

## 🎯 Objectif
Développer une solution capable de :
1.  **Détecter** la position d'une plaque d'immatriculation dans une image (LP Detection).
2.  **Lire** les caractères présents sur la plaque (OCR).

## 🛠️ Approche Technique
Le projet repose sur un fine-tuning du modèle **YOLOv8** (nano), et non un entraînement complet à partir de zéro (Transfer Learning).

### Détails de l'entraînement
*   **Modèle de base** : `yolov8n.pt` (pré-entraîné sur COCO).
*   **Transfert** : 319/355 couches transférées.
*   **Configuration** :
    *   Époques : 100
    *   Taille d'image : 320x320 pixels
    *   Optimiseur : AdamW (lr0=0.002)

### Pipeline d'inférence
Le notebook `ALPR_YOLOv8_FastPlateOCR.ipynb` orchestre deux tâches :
1.  **Détection** : Localisation des plaques avec le modèle fine-tuné.
2.  **OCR** : Reconnaissance des caractères via la librairie spécialisée `fast-plate-ocr` (modèle global entraîné sur 220k+ plaques).

## 📊 Données et Provenance
Les données utilisées pour l'entraînement proviennent du dataset public **UC3M-LP**.

*   **Source** : Dataset académique de l'Université Carlos III de Madrid (UC3M), hébergé sur Roboflow Universe.
*   **Contenu** : Images de véhicules capturées sur le campus universitaire pour la recherche sur les systèmes de transport intelligents (ITS).
*   **Licence** : **CC BY 4.0** (Creative Commons Attribution 4.0 International).
    *   *Crédits : Roboflow Universe Projects / license-plate-recognition-rxg4e / UC3M.*

## ⚖️ RGPD et Éthique
La collecte initiale des données et le consentement des personnes relèvent de la responsabilité de l’organisme ayant constitué le dataset (Université Carlos III de Madrid).

Ce projet s’inscrit strictement en tant que **réutilisation de données publiques à des fins pédagogiques et de recherche**. Le traitement réalisé est limité à la détection de plaques, sans identification des personnes ni conservation des données sensibles, conformément aux principes de minimisation et de limitation de finalité du RGPD.

## 🚀 Résultats
Le modèle fine-tuné démontre une capacité robuste à détecter les plaques dans des conditions variées (jour/nuit, angles différents). Les résultats de l'OCR sont consolidés par l'utilisation d'une librairie spécialisée post-détection.