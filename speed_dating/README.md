# Speed Dating Analysis (Tinder)

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=fff)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Quelles sont les motivations qui incitent les individus à envisager un second rendez-vous ensemble ?  
Ce projet explore un jeu de données rassemblant des informations sur des milliers de rencontres organisées auprès d'étudiants, pour comprendre ce qui déclenche l'envie de se revoir.

> **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp.

---

## Le projet en quelques mots

Le but de cette Analyse Exploratoire des Données (EDA) est de fournir des insights "actionnables" à l'équipe marketing d'une application de rencontre (type Tinder). L'analyse cherche à identifier :
- Les biais cognitifs (différence entre l'auto-évaluation et l'évaluation reçue).
- L'importance réelle de l'attractivité physique (souvent minorée dans les déclarations par rapport aux actes).
- L'impact de la fatigue décisionnelle (la probabilité de dire "oui" baisse au fur et à mesure des rencontres).

---

## Les sources de données

L'analyse repose sur un jeu de données robuste de **7466 rencontres** (après nettoyage) impliquant 551 participants uniques.
- Fichier source brut nettoyé : `SpeedDating_Cleaned.csv`
- Les données ont été préalablement standardisées et filtrées des valeurs aberrantes ou des rencontres incomplètes via le script de nettoyage.

---

## Installation

```bash
cd "speed_dating"
pip install pandas matplotlib seaborn jupyter
```

---

## Les notebooks et fichiers

| Fichier | Ce qu'il fait |
|---|---|
| `cleaning.py` | Script de nettoyage des données brutes (suppression des NA, filtrage, etc.) |
| `Speed_Dating_Analysis_Cleaned.ipynb` | Le notebook Jupyter principal contenant l'EDA et les visualisations |
| `Presentation_Speed_Dating.md` | Résumé des conclusions (avec graphiques générés) |
| `FAQ.md` | Questions/Réponses pour la soutenance orale |

---

## Résultats et Recommandations

L'analyse des données a permis de mettre en lumière des comportements clés et de formuler plusieurs recommandations :

1. **Gérer l'écart de perception** : Aider les utilisateurs à mieux se présenter pour réduire le décalage entre leur auto-évaluation et la perception des autres (ex: conseils sur le choix des photos).
2. **Limiter la fatigue décisionnelle** : La probabilité d'accepter un rendez-vous chute après 10-15 profils. Mettre un plafond de rencontres par session pourrait accroître la qualité de l'attention.
3. **L'importance du visuel** : L'attirance physique est le critère n°1 pour déclencher un "match", même si les utilisateurs déclarent chercher d'abord l'intelligence ou la sincérité. L'algorithme doit privilégier les affinités visuelles plutôt que la simple compatibilité des centres d'intérêt.

---

## Structure du projet

```text
speed_dating/
├── src/                                     # Images générées pour la présentation
├── cleaning.py                              # Nettoyage initial des données
├── Speed_Dating_Analysis_Cleaned.ipynb      # Notebook complet d'analyse
├── SpeedDating_Cleaned.csv                  # Dataset final
├── Presentation_Speed_Dating.md             # Synthèse et conclusions
├── FAQ.md                                   # FAQ pour l'oral
└── README.md                                # Ce fichier
```

---

## Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
