# The North Face E-commerce - NLP & Système de Recommandation 🏔️

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![SpaCy](https://img.shields.io/badge/SpaCy-09A3D5?style=flat&logo=spacy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Le département marketing de **The North Face** souhaite exploiter le Machine Learning pour augmenter le taux de conversion sur sa boutique en ligne. 

> ⚠️ **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp.

---

## 📖 Le projet en quelques mots

Le catalogue actuel n'est pas structuré de façon optimale et les clients ont du mal à trouver des produits similaires. Notre mission a consisté à analyser 500 descriptions de produits pour :
1. Construire un **système de recommandation** ("Vous aimerez aussi...") basé sur la similarité des descriptions.
2. Extraire la structure sous-jacente du catalogue (**Topic Modeling** et **Clustering**) pour réorganiser les catégories de manière algorithmique.

Il s'agit d'un projet d'**Apprentissage Non Supervisé** (sans étiquettes prédéfinies).

---

## 📊 Les sources de données

- **Source** : Dataset Kaggle `sample-data.csv` contenant 500 descriptions de produits The North Face.
- **Défis** : Texte brut avec balises HTML, présence massive de mots vides, textes de longueurs très variables (de 400 à 3500 caractères).

---

## ⚙️ Installation

```bash
cd "The North Face ecommerce"
pip install pandas scikit-learn spacy
python -m spacy download en_core_web_sm
```

---

## 🚀 Les notebooks et fichiers

| Fichier | Ce qu'il fait |
|---|---|
| `the_north_face_fr.ipynb` | Notebook complet d'analyse, de nettoyage NLP et de clustering (VF) |
| `the_north_face_en.ipynb` | Notebook complet d'analyse (Version anglaise) |
| `Presentation_The_North_Face.md` | Synthèse métier et explication des algorithmes non supervisés |
| `discours.md` | Script d'accompagnement pour la soutenance orale |
| `FAQ.md` | Foire aux questions technique |

---

## 🏗️ Architecture et Modélisation (NLP)

### 1. Le Preprocessing du Texte (Pipeline NLP)
- **Nettoyage HTML et Alpha** : Suppression des balises (`<br>`) et des chiffres.
- **SpaCy** : Tokenisation, lemmatisation (ex: "wicking" -> "wick") et suppression de plus de 300 mots vides anglais.
- **Vectorisation** : Transformation du texte en matrice mathématique via `TfidfVectorizer` en capturant les mots uniques et les bigrammes (ex: "organic cotton").

### 2. Le Clustering (DBSCAN)
- L'algorithme `DBSCAN` a été privilégié par rapport à K-Means car il détecte automatiquement le nombre de clusters, gère les formes arbitraires et permet d'isoler les produits uniques comme "outliers" (bruit).
- Résultat : Identification de micro-catégories pertinentes (ex: pantalons techniques, jeans bio, brassières).

### 3. Topic Modeling (LSA / TruncatedSVD)
- Réduction de dimension permettant d'isoler 15 "Sujets Latents" principaux.
- Le modèle a notamment révélé que le discours *éco-responsable* domine le catalogue, tout en découvrant automatiquement une catégorie inattendue (la vente de "Posters" déco !).

### 4. Recommender System
- Création d'un moteur de similarité basé sur la **Distance Cosinus** au sein des clusters ou globalement. Le système renvoie instantanément le "Top-5" des produits les plus similaires sémantiquement.

---

## 📂 Structure du projet

```text
The North Face ecommerce/
├── sample-data.csv                    # Dataset brut
├── the_north_face_fr.ipynb            # Notebook principal
├── the_north_face_en.ipynb            # Version anglaise du notebook
├── wordclouds_*.png                   # Visualisations générées
├── Presentation_The_North_Face.md     # Synthèse du projet
├── discours.md                        # Trame pour la présentation
├── FAQ.md                             # Anticipation Q&A
└── README.md                          # Ce fichier
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
