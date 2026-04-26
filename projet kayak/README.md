# KAYAK - Plan Your Trip with Data ✈️

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonaws&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![Scrapy](https://img.shields.io/badge/Scrapy-60A5FA?style=flat&logo=scrapy&logoColor=fff)](#)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=fff)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Vous vous êtes déjà demandé où partir ce week-end en étant sûr d'avoir du soleil et un bon hôtel ? C'est pour répondre à cette question — à l'échelle de 35 villes françaises — que j'ai construit ce pipeline de données.

> ⚠️ **Note :** Les images présentes dans ce document sont des visuels d'illustration. Elles ne sont pas générées directement par l'exécution des notebooks Jupyter.

![Hero](img_hero.png)

---

## 📖 Le projet en quelques mots

Ce projet est un exercice de **Data Engineering de bout en bout**, réalisé dans le cadre de la formation Data Fullstack chez JEDHA Bootcamp. Le brief (fourni dans `01-Plan_your_trip_with_Kayak.ipynb`) demandait de :

- Collecter les données météo de 35 villes françaises via des APIs.
- Scraper les informations hôtelières sur Booking.com.
- Stocker le tout dans une architecture Cloud AWS (S3 → RDS PostgreSQL).
- Produire des cartes interactives de recommandation.

L'architecture, les outils et les sources de données étaient largement guidés par l'énoncé. Mon travail a consisté à implémenter ce plan et à m'adapter quand la réalité du terrain s'écartait de la théorie (spoiler : Booking.com n'aime pas les scrapers).

---

## 📊 Les sources de données

Le pipeline s'appuie sur trois sources, toutes suggérées par le brief :

- **Nominatim (OpenStreetMap)** : Géocodage des 35 villes, puis des hôtels (cette deuxième utilisation est un ajout personnel pour compenser l'impossibilité de scraper les coordonnées GPS directement sur Booking).
- **OpenWeatherMap (API Forecast Standard v2.5)** : Prévisions météo à 5 jours par blocs de 3 heures.
- **Booking.com (Web Scraping via Scrapy)** : Extraction des données hôtelières en contournant les protections AWS WAF via injection de cookies de session.

---

## ⚙️ Installation

```bash
cd "projet kayak"
pip install pandas sqlalchemy psycopg2-binary plotly boto3 scrapy
```

Les clés API (OpenWeatherMap) et les credentials AWS doivent être configurés avant d'exécuter le pipeline.

---

## 🚀 Les notebooks du pipeline

Le projet est découpé en notebooks indépendants. Ce découpage n'est pas parfait (l'ETL est répartie sur deux fichiers au lieu d'un seul, par exemple), mais chaque brique est autonome et réexécutable séparément.

| Notebook | Ce qu'il fait |
|---|---|
| `meteo api booking v3.ipynb` | Appels API OpenWeatherMap + calcul du score météo |
| `booking avec description.ipynb` | Scraping Booking.com (avec contournement anti-bot) |
| `geocodageafterscrap.ipynb` | Géocodage GPS des hôtels via Nominatim (adaptation non prévue par le brief) |
| `upload.ipynb` | Upload des CSV bruts vers Amazon S3 |
| `etlbooking.ipynb` | ETL Hôtels : nettoyage Regex + chargement RDS |
| `meteo et cartes optimisees.ipynb` | ETL Météo + génération des cartes Plotly |

---

## 🏗️ Architecture

L'architecture Data Lake → Data Warehouse était demandée par l'énoncé. Je l'ai implémentée comme suit :

```text
Nominatim API ──┐
OpenWeatherMap ──┼──→ Python (Pandas) ──→ CSV bruts ──→ AWS S3 (Data Lake)
Booking.com ────┘                                            │
(Scrapy)                                                     ▼
                                                  ETL (Nettoyage + Regex)
                                                             │
                                                             ▼
                                                  AWS RDS PostgreSQL (Warehouse)
                                                             │
                                                             ▼
                                                Plotly (Cartes carto-positron)
```

- **S3** : Stockage brut et persistant. Indispensable vu la fragilité du scraping — les cookies expirent, la structure HTML peut changer du jour au lendemain.
- **RDS** : Données nettoyées, typées, prêtes à être requêtées en SQL.

Pour aller plus loin, consultez la [Synthèse du projet](Synthese_Projet_Kayak.md) et la [FAQ technique](FAQ.md).

---

## 💰 Coûts

| Composant | Phase projet | Estimation production |
|-----------|-------------|------------|
| **AWS S3** | Gratuit (Free Tier) | ~0.023 $/Go/mois |
| **AWS RDS** | Gratuit (Free Tier) | Variable selon instance |
| **Nominatim** | Gratuit | Gratuit |
| **OpenWeatherMap** | Gratuit (API Standard) | Gratuit dans les limites |
| **Total projet** | **0 €** | — |

---

## 🛡️ RGPD & Éthique

- Aucune donnée personnelle collectée — uniquement des informations publiques sur les hôtels et la météo.
- Scraping avec délais entre les requêtes et headers réalistes, conformément aux bonnes pratiques.
- Secrets (clés API, credentials AWS) non commités dans le dépôt.

---

## 🗺️ Résultats & Visualisations

> ⚠️ **Rappel :** Les captures ci-dessous sont des **images d'illustration** pour montrer le type de cartes que le pipeline produit. Ce ne sont pas les exports bruts des notebooks.

*Note technique : les cartes utilisent le style `carto-positron` au lieu d'OpenStreetMap, pour contourner des problèmes d'affichage liés à la Referrer Policy dans Jupyter.*

### Top 5 des Destinations
Score calculé via : `Température Moyenne − (Pluie Totale en mm × 2)`

![Carte Météo](img_weather.png)

### Inventaire Hôtelier
699 hôtels géolocalisés sur les 880 scrapés.

![Carte Hôtels](img_hotels.png)

### Architecture
![Architecture](img_architecture.png)

---

## 📂 Structure du projet

```text
projet kayak/
├── 01-Plan_your_trip_with_Kayak.ipynb   # Brief du projet (énoncé JEDHA)
├── meteo api booking v3.ipynb           # API Météo
├── booking avec description.ipynb       # Scraper Booking
├── geocodageafterscrap.ipynb            # Géocodage post-scraping
├── upload.ipynb                         # Upload S3
├── etlbooking.ipynb                     # ETL Hôtels
├── meteo et cartes optimisees.ipynb     # ETL Météo + Cartes
├── *.csv                                # Données
├── img_*.png                            # Illustrations
├── Synthese_Projet_Kayak.md             # Retour d'expérience détaillé
├── FAQ.md                               # Questions techniques fréquentes
└── README.md                            # Ce fichier
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
