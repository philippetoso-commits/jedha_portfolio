# Projet Steam - Analyse EDA avec PySpark & Databricks

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat&logo=apachespark&logoColor=fff)](#)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

**Client fictif** : Ubisoft  
**Objectif** : Comprendre l'ecosysteme du jeu video sur Steam pour guider la strategie d'un nouveau jeu revolutionnaire.

---

## Le projet en quelques mots

Ce projet consiste a realiser une Analyse Exploratoire des Donnees (EDA) a grande echelle en utilisant un environnement Big Data. Ubisoft souhaite analyser le marche mondial du jeu video sur Steam pour identifier les tendances actuelles et les facteurs de popularite.

L'analyse est segmentee en trois grands niveaux :
- **Analyse Macro** (editeurs, prix, annees de sortie, langues).
- **Analyse des Genres** (ratio d'avis positifs, lucrativite par genre).
- **Analyse de la Plateforme** (disponibilite selon les OS).

---

## Les sources de donnees

Le projet s'appuie sur un dataset semi-structure (JSON imbrique) stocke dans un Data Lake AWS S3 :
- **Source** : `s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json`
- **Contenu** : Des dizaines de milliers de jeux avec leurs metadonnees (prix, editeur, date de sortie, notes positives/negatives, genres, categories, etc.).

---

## Installation et execution

L'ensemble de l'analyse est contenu dans un notebook compatible Databricks.

1. Ouvrez votre workspace Databricks.
2. Allez dans **Workspace -> Import**.
3. Importez le fichier `steam_eda.py` (Databricks le reconnait automatiquement comme un notebook grace aux balises de commentaires).
4. Attachez un cluster Spark et cliquez sur **Run All**.

## Lien du notebook Databricks

Notebook du projet Steam (Databricks) :  
[Ouvrir le notebook Databricks](https://dbc-bb6c41cb-1929.cloud.databricks.com/editor/notebooks/1658891342594298?o=3029848363093215)

---

## Architecture et Stack technique

L'environnement impose l'utilisation des technologies Big Data :

```text
AWS S3 (Data Lake JSON) ──> Databricks Cluster ──> PySpark (DataFrames) ──> Visualisations (display)
```

- **PySpark** : Manipulation performante du JSON. Utilisation intensive de `explode_outer()` pour aplanir les tableaux (ex: genres) et `getField()` pour acceder aux structures imbriquees (ex: prix).
- **Databricks** : Environnement d'execution et outil de visualisation natif via la commande `display()`.

---

## Structure du projet

```text
projet steam/
├── steam_eda.py      <- Notebook PySpark complet (format Databricks .py)
└── README.md         <- Ce fichier
```

---

## Auteur
Projet realise par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
