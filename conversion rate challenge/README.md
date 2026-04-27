# Conversion Rate Challenge - Machine Learning & Ensembling 🏆

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-1793D1?style=flat&logo=xgboost&logoColor=fff)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-0080FF?style=flat&logo=lightgbm&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Ce dossier représente l'aboutissement d'une compétition (Hackathon / Challenge de type Kaggle) visant à prédire avec la plus grande précision possible le **taux de conversion** des utilisateurs d'un site web. 

> ⚠️ **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp et s'apparente à un "Laboratoire de R&D" intensif.

---

## 📖 Le projet en quelques mots

Le but est d'optimiser le taux de conversion (Conversion Rate Optimization - CRO) à l'aide d'algorithmes prédictifs. La tâche principale est de classer les utilisateurs (Vont-ils convertir ou non ?) en se basant sur leur comportement de navigation, leur provenance et leurs caractéristiques géographiques/démographiques.

Contrairement aux autres projets, l'objectif ici n'était pas seulement de construire *un* modèle, mais de trouver le modèle mathématiquement imbattable en explorant toutes les techniques avancées d'apprentissage supervisé (Feature Engineering agressif, Hyperparameter Tuning, et Ensembling).

---

## 📊 Les données et l'Exploration

- **Données d'entraînement** : `conversion_data_train.csv`
- **Données de test (pour la soumission)** : `conversion_data_test.csv`

**Analyse Exploratoire (EDA)** : 
Le rapport généré `eda_conversion_rate_report.html` permet de comprendre les comportements : par exemple, le nombre de pages visitées est un indicateur extrêmement fort (les utilisateurs visitant plus de 10 pages ont une probabilité presque garantie d'acheter).

---

## ⚙️ Installation

```bash
cd "conversion rate challenge"
pip install pandas scikit-learn xgboost lightgbm catboost optuna
```

---

## 🚀 La Modélisation et le "Laboratoire"

L'ampleur du dossier (plus de 200 scripts et fichiers) témoigne d'une approche itérative et agressive pour grappiller chaque fraction de précision (F1-Score / Accuracy) sur le Leaderboard.

### Les stratégies déployées :
1. **Modèles isolés** : Tests exhaustifs de Régression Logistique, Random Forest, Gradient Boosting, XGBoost, CatBoost et réseaux de neurones simples.
2. **Feature Engineering ("Divination / Features")** : Création de dizaines de nouvelles variables (polynômes, ratios, croisements de variables) pour aider les arbres de décision.
3. **Optimisation Mathématique ("Optuna")** : Utilisation d'algorithmes de recherche bayésienne pour ajuster automatiquement les hyperparamètres complexes des modèles.
4. **Ensembling & Syndicat ("Le Sénat / Pirate / Monster")** : Fusion des prédictions de plusieurs modèles de natures différentes (ex: Voting Classifier, Stacking) pour annuler les faiblesses individuelles de chaque modèle ("Sagesse de la foule").

### Exemples d'expérimentations notables incluses :
- `Simulation_Gradient_Boosting.ipynb`
- `Optuna_Champion_Implementation.ipynb`
- Scripts de "Syndicate" et "Audits" pour analyser les *False Negatives* ou la calibration des probabilités.

---

## 📂 Structure du projet

Ce dossier fonctionne comme un "Banc d'essai". Voici les grandes catégories de fichiers :

```text
conversion rate challenge/
├── conversion_data_train.csv          # Dataset d'entraînement
├── eda_conversion_rate_report.html    # Rapport d'analyse métier des datas
├── conversion_rate.ipynb              # Notebook initial d'approche (Baseline)
├── run_*.py / analyze_*.py            # Scripts de modélisation, d'audit et d'ensembling
├── *_predictions.csv                  # Multiples soumissions générées pour la compétition
├── README.md                          # Ce fichier
└── ... (plusieurs notebooks d'expérimentation R&D)
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
