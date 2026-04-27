# Getaround - Pricing & Delay Analysis 🚗

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=fff)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=fff)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=000)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

Ce projet est conçu pour aider **Getaround** (le leader européen de l'autopartage) à résoudre deux problématiques majeures : l'optimisation des prix de location et la gestion des retards au moment du check-out. 

> ⚠️ **Note :** Ce projet a été réalisé dans le cadre de la certification "Concepteur Développeur en Science des Données" (Bloc 5) chez JEDHA Bootcamp.

---

## 📖 Le projet en quelques mots

L'objectif principal est de développer un produit Data de bout en bout comprenant :
- Un **Dashboard interactif (Streamlit)** permettant aux Product Managers d'analyser l'impact d'un seuil minimum de délai entre deux locations pour réduire les frictions liées aux retards, tout en minimisant la perte de revenus.
- Une **API de Machine Learning (FastAPI)** capable de prédire le prix de location optimal par jour pour une voiture donnée en fonction de ses caractéristiques (modèle, kilométrage, type de carburant, options, etc.).

---

## 📊 Les sources de données

Le projet s'appuie sur deux jeux de données fournis :
- **Getaround Delay Analysis (`get_around_delay_analysis.csv`)** : Contient l'historique des locations, incluant l'état du check-in, le retard au check-out en minutes, et le délai avec la location précédente.
- **Getaround Pricing (`get_around_pricing_project.csv`)** : Contient les caractéristiques de milliers de voitures louées sur la plateforme ainsi que leur prix de location journalier.

---

## ⚙️ Installation

```bash
cd "Analyse Getaround"
pip install -r requirements.txt # si disponible, ou installez manuellement
pip install pandas scikit-learn fastapi uvicorn streamlit
```

---

## 🚀 Les livrables et l'architecture

Le projet est divisé en plusieurs briques déployées dans le cloud :

### 1. Dashboard Streamlit (Analyse des retards)
- **[Lien du Dashboard](https://philippetos-getaround.hf.space)** (Hébergé sur Hugging Face Spaces)
- Permet de simuler l'impact de différents seuils (par ex. 60 min, 120 min) sur les frictions résolues vs le revenu perdu.

### 2. API FastAPI (Prédiction de prix)
- **[Swagger UI de l'API](https://philippetos-getaroundapi.hf.space/docs)** (Hébergé sur Hugging Face Spaces)
- Expose un modèle de Machine Learning entraîné pour prédire le prix.
  
*Exemple de requête :*
```bash
curl -i -H "Content-Type: application/json" -X POST -d '{
  "model_key": "Renault",
  "mileage": 50000,
  "engine_power": 120,
  "fuel": "diesel",
  "paint_color": "grey",
  "car_type": "sedan",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": true,
  "automatic_car": false,
  "has_getaround_connect": false,
  "has_speed_regulator": true,
  "winter_tires": true
}' https://philippetos-getaroundapi.hf.space/predict
```

---

## 🏗️ Architecture du Déploiement

```text
Data Brute (CSV) ──┐
                   ├─→ Python (Jupyter) ──→ Modèles Machine Learning (.pkl)
Analyse (EDA) ─────┘                                    │
                                                        ▼
                    ┌────────────────────────────────────────────────────────┐
                    │                      Hugging Face                      │
                    │                                                        │
                    │   ┌────────────────────┐      ┌────────────────────┐   │
                    │   │   Space Streamlit  │      │   Space FastAPI    │   │
                    │   │    (Dashboard)     │      │   (API Pricing)    │   │
                    │   └────────────────────┘      └────────────────────┘   │
                    └────────────────────────────────────────────────────────┘
```

---

## 📂 Structure du projet

```text
Analyse Getaround/
├── hf_dashboard_repo/                   # Code source du Dashboard Streamlit
│   ├── app.py
│   ├── Dockerfile
│   └── get_around_delay_analysis.csv
├── hf_api_repo/                         # Code source de l'API FastAPI
│   ├── app.py
│   ├── Dockerfile
│   └── model.pkl (ou équivalent)
├── notebooks/                           # Notebooks d'exploration et d'entraînement ML
│   ├── GetAround_EDA_ML_EN.ipynb
│   └── GetAround_EDA_ML_FR.ipynb
├── Presentation_GetAround.md            # Slides de présentation
├── discours.md                          # Script pour l'oral
├── FAQ.md                               # Questions/Réponses préparées
└── README.md                            # Ce fichier
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
