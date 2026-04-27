# Uber Pickups - Recommandation de Zones Chaudes 🚕

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Le temps d'attente est le premier facteur d'annulation d'une course. Ce projet vise à résoudre ce point de friction majeur pour **Uber** en prédisant et recommandant les "Hot-Zones" (zones de forte demande) aux chauffeurs avant même que la demande n'arrive.

> ⚠️ **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science chez JEDHA Bootcamp.

---

## 📖 Le projet en quelques mots

Afin d'éviter que les clients attendent plus de 5 à 7 minutes, nous avons construit un système d'Intelligence Artificielle en deux étapes :
1. **Apprentissage Non Supervisé (Clustering)** : Découpage intelligent et dynamique de l'espace urbain de New York City en zones d'activité réelles (K-Means / DBSCAN) plutôt qu'un simple quadrillage arbitraire.
2. **Apprentissage Supervisé (Classification)** : Entraînement d'un modèle Machine Learning (Random Forest) capable de dire au chauffeur dans quelle Hot-Zone il devrait se diriger *maintenant*, en fonction de l'heure et du jour de la semaine.

---

## 📊 Les sources de données

Le projet utilise un extrait du jeu de données historique public des courses Uber à New York :
- **Uber Trip Data (Avril 2014)** : Historique des prises en charge incluant la date, l'heure, la latitude et la longitude exactes de la commande.

---

## ⚙️ Installation

```bash
cd "Uber_Pickups"
# Décompressez l'archive de données (si nécessaire)
unzip uber.zip 
pip install pandas scikit-learn plotly jupyter
```

---

## 🚀 Les notebooks et fichiers

Le projet est documenté et structuré via les fichiers suivants :

| Fichier | Ce qu'il fait |
|---|---|
| `uber_pickups_fr.ipynb` | Notebook complet d'Analyse (EDA) et de modélisation (VF) |
| `uber_pickups_en.ipynb` | Notebook complet d'Analyse (EDA) et de modélisation (EN) |
| `uber_hotzones_kmeans_fr.html` | Export HTML interactif des visualisations spatiales (Plotly) |
| `Presentation_Uber_Pickups.md` | Synthèse métier et résultats des algorithmes |
| `discours.md` | Script d'accompagnement pour la soutenance orale |
| `FAQ.md` | Foire aux questions technique |

---

## 🏗️ Architecture et Modélisation

Le flux de traitement Machine Learning repose sur un Pipeline Scikit-Learn robuste :

1. **Feature Engineering** : Extraction du *Jour* et de *l'Heure* à partir des données temporelles brutes.
2. **Clustering** : Exécution de `K-Means` pour déterminer 10 clusters (Hot-Zones) sur la carte de Manhattan, servant de labels cibles. Expérimentation avec `DBSCAN` pour cibler les artères saturées par densité.
3. **Preprocessing** : Encodage `OneHotEncoder` pour les catégories et `StandardScaler` pour équilibrer le poids de l'Heure (0-24) vs le Jour (0-7).
4. **Moteur Prédictif** : Un modèle `RandomForestClassifier` combinant 50 arbres de décision pour une prédiction non-linéaire optimale.

---

## 💡 Résultats et Explicabilité

L'analyse de l'importance des variables (*Feature Importance*) issue du Random Forest a démontré de manière mathématique le comportement de la demande :
- **L'Heure (Hour)** : Dirige à plus de **63%** le comportement de commande.
- **Le Jour (DayOfWeek)** : Pèse pour environ **30%**, confirmant la bascule des trajets professionnels de semaine vers les zones de loisirs/bars le week-end.

En fournissant une stratégie temporelle claire, cet algorithme permet aux chauffeurs Uber de se positionner proactivement, maximisant ainsi les revenus et la satisfaction client.

---

## 📂 Structure du projet

```text
Uber_Pickups/
├── uber.zip                             # Dataset brut (Lat, Lon, Date)
├── uber_pickups_fr.ipynb                # Notebook principal
├── uber_pickups_en.ipynb                # Version anglaise du notebook
├── uber_hotzones_kmeans_fr.html         # Carte interactive générée
├── Presentation_Uber_Pickups.md         # Synthèse du projet
├── discours.md                          # Trame pour la présentation
├── FAQ.md                               # Anticipation Q&A
└── README.md                            # Ce fichier
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
