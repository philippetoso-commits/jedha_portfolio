# Walmart - Détecteur de Ventes et Analyse Prédictive
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Estimer avec précision les ventes hebdomadaires des différents magasins est vital pour optimiser la gestion des stocks et la planification du personnel. Ce projet s'attaque à ce défi pour le géant américain de la grande distribution : **Walmart**.

> **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp.

---

## Le projet en quelques mots
Plutôt que de se baser sur de simples statistiques historiques ou sur l'instinct, l'objectif est de construire un modèle d'Intelligence Artificielle (Machine Learning) robuste. Ce modèle prend en compte des indicateurs externes (comme la température, le prix du carburant, le taux de chômage ou les jours fériés) pour **prédire de façon automatique le chiffre d'affaires hebdomadaire**. 

Le défi principal du projet réside dans la gestion du **surapprentissage (overfitting)** afin de s'assurer que le modèle puisse généraliser ses prédictions pour l'avenir.

---

## Les sources de données
Le projet exploite l'historique de ventes (*Weekly_Sales*) de différents magasins Walmart (dataset : `Walmart_Store_sales.csv`). 

L'Analyse Exploratoire des Données (EDA) a mis en évidence plusieurs points clés qui ont nécessité un nettoyage rigoureux :
- Gestion des valeurs manquantes.
- Extraction temporelle (Jour, Mois, Année) à partir des dates brutes.
- Écrémage des "outliers" (valeurs aberrantes à plus de 3 écarts-types) pour la température, le chômage, etc.

---

## Installation
```bash
cd "Wallmart"
pip install pandas scikit-learn jupyter
```

---

## Livrables
- **Notebook principal en français** : `walmart_sales_fr.ipynb`
- **Notebook en anglais** : `walmart_sales_en.ipynb`
- **Code source complet (GitHub)** : https://github.com/philippetoso-commits/jedha_portfolio
- **Lien direct dossier projet** : https://github.com/philippetoso-commits/jedha_portfolio/tree/main/Wallmart

---

## Les notebooks et fichiers
Le projet est documenté et structuré via les fichiers suivants :

| Fichier | Ce qu'il fait |
|---|---|
| `walmart_sales_fr.ipynb` | Notebook complet contenant l'EDA, le Preprocessing et la Modélisation (VF) |
| `walmart_sales_en.ipynb` | Notebook complet d'analyse (Version anglaise) |
| `Presentation_Walmart.md` | Synthèse métier des approches (Ridge, Lasso) et résultats |
| `discours.md` | Script d'accompagnement pour la soutenance orale |
| `FAQ.md` | Foire aux questions technique |

---

## Architecture et Modélisation
Le flux d'apprentissage supervisé a été construit de manière méthodique :

### 1. Preprocessing (Scikit-Learn Pipeline)
- Remplacement des valeurs vides via `SimpleImputer`.
- Standardisation des variables numériques via `StandardScaler` (essentiel pour mettre la température et le taux de chômage sur une échelle mathématique équitable).
- Binarisation des variables catégorielles (ex: jour férié) via `OneHotEncoder`.

### 2. Modèles Testés et Régularisation
Plutôt que de se contenter d'une simple ligne droite, nous avons testé 3 approches pour lisser la dépendance aux variables historiques :
- **Baseline (Régression Linéaire)** : Excellent score d'entraînement (R² = 0.977) mais léger surapprentissage détecté en test.
- **Le Modèle Ridge (L2)** : Une régression "bridée" qui pénalise doucement les variables qui prennent trop d'importance, forçant le modèle à équilibrer son apprentissage (Score R² = 0.892).
- **Le Modèle Lasso (L1)** : Une régression "sévère" qui procède à la sélection automatique de caractéristiques en forçant les coefficients des données inutiles à zéro (Score R² = 0.897, **meilleur modèle**).

### 3. Optimisation (GridSearch)
Les paramètres optimaux (ex: force de la pénalité `alpha` pour Lasso/Ridge) n'ont pas été choisis au hasard, mais trouvés de façon exhaustive grâce à un `GridSearchCV`.

---

## Structure du projet
```text
Wallmart/
├── Walmart_Store_sales.csv            # Dataset brut
├── walmart_sales_fr.ipynb             # Notebook principal
├── walmart_sales_en.ipynb             # Version anglaise du notebook
├── Presentation_Walmart.md            # Synthèse du projet
├── discours.md                        # Trame pour la présentation
├── FAQ.md                             # Anticipation Q&A
└── README.md                          # Ce fichier
```

---

## Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
