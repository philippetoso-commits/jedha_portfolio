# Conversion Rate Challenge - Machine Learning & Ensembling

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-1793D1?style=flat&logo=xgboost&logoColor=fff)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-0080FF?style=flat&logo=lightgbm&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Ce dossier représente l'aboutissement d'une compétition (Hackathon / Challenge de type Kaggle) visant à prédire avec la plus grande précision possible le **taux de conversion** des utilisateurs d'un site web. 

> **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp et s'apparente à un "Laboratoire de R&D" intensif.

---

---

## Le Champion : Stratégie "ADN" (Single Model Theory)

Le modèle ayant remporté la **1ère place du classement** est basé sur la stratégie dite de "l'ADN du dataset" (implémentée dans  `ADN_Optimise_Final.ipynb`).

### Performance Leaderboard
| Modèle | F1-Score | Précision | Recall |
| :--- | :--- | :--- | :--- |
| **submission_adn_tosophilippe** | **0.7658** | **0.8002** | **0.7343** |

> "Hello Philippe, bien joué, t'as pris la tête du classement" — Feedback instructeur.

### Fonctionnement technique
Plutôt que d'empiler des modèles complexes (Ensembling), cette approche repose sur la **pureté mathématique** et la physique du dataset :
*   **Moteur** : Un unique modèle **XGBoost** (Bernoulli MLE).
*   **Contraintes de Monotonie** : Application de contraintes strictes `(0, 0, 0, -1, 1)` — l'âge doit mathématiquement faire baisser la probabilité de conversion, tandis que le nombre de pages visitées doit la faire augmenter. Cela empêche le modèle d'apprendre du "bruit" statistique.
*   **Optimisation de Seuil** : Recherche du seuil de probabilité optimal (trouvé à **0.385**) au lieu du standard 0.5, pour maximiser spécifiquement le F1-Score.

---

## Les données et l'Exploration

- **Données d'entraînement** : `conversion_data_train.csv`
- **Données de test (pour la soumission)** : `conversion_data_test.csv`

**Analyse Exploratoire (EDA)** : 
Le rapport généré `eda_conversion_rate_report.html` permet de comprendre les comportements : par exemple, le nombre de pages visitées est un indicateur extrêmement fort (les utilisateurs visitant plus de 10 pages ont une probabilité presque garantie d'acheter).

---

## Installation

```bash
cd "conversion rate challenge"
pip install pandas scikit-learn xgboost lightgbm catboost optuna
```

---

## La Modélisation et le "Laboratoire" (R&D)
 
Ce dossier présente les approches les plus performantes issues d'un laboratoire de R&D ayant exploré plus de 200 scripts et configurations. Si seuls les modèles "finalistes" sont présentés ici, l'ensemble des techniques de Data Science moderne a été mobilisé pour maximiser le F1-Score.
 
### Les stratégies explorées :
1. **Modèles isolés** : Tests de Régression Logistique, Random Forest, CatBoost et réseaux de neurones.
2. **Exploration intensive de l'Ensembling** : Fusion de modèles (Voting, Stacking) pour tester la "sagesse de la foule".
3. **Optimisation Mathématique ("Optuna")** : Recherche bayésienne pour le réglage fin des hyperparamètres.
4. **Physique des données (Approche ADN)** : Simplification radicale vers un modèle XGBoost monotone, qui s'est avérée être la stratégie gagnante.

---

## Structure du projet

Ce dossier contient les modèles "finalistes" issus d'une phase de recherche intensive. 

```text
conversion rate challenge/
├── conversion_data_train.csv        # Dataset d'entraînement
├── eda_conversion_rate_report.html  # Rapport d'analyse métier (EDA)
├── ADN_Optimise_Final.ipynb         # LE CHAMPION (936 conversions) - Stratégie ADN
├── HistGradientBoostingClassifier.ipynb # MODÈLE SNIPER (922 conversions)
├── conversion_rate.ipynb            # Notebook Baseline (Approche initiale)
├── submission_adn_tosophilippe.csv  # Soumission gagnante (Top 1 Leaderboard)
└── README.md                        # Ce fichier
```

---

## Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
