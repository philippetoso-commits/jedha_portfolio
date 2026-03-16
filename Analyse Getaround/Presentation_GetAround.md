# GetAround — Présentation Projet
## Jedha Bootcamp — Bloc 5 — Industrialisation d'un Algorithme ML

---

## Slide 1 — Introduction

**Projet : GetAround**

> *"L'Airbnb des voitures"*

- Fondée en **2009** — +5 millions d'utilisateurs, ~20 000 voitures disponibles
- Modèle : location de voiture entre particuliers, de quelques heures à plusieurs jours
- Partenaire officiel **Jedha Bootcamp**

**Objectifs du projet :**
1.  Analyser l'impact des **retards de restitution** et recommander un délai minimum
2.  Construire un **modèle ML** pour prédire le prix journalier optimal d'un véhicule

---

## Slide 2 — Les Données

### Dataset 1 — Delay Analysis (`get_around_delay_analysis.xlsx`)
| Colonne | Description |
|---------|-------------|
| `rental_id` | Identifiant de la location |
| `car_id` | Identifiant du véhicule |
| `checkin_type` | `mobile` ou `connect` |
| `state` | `ended` ou `canceled` |
| `delay_at_checkout_in_minutes` | Retard à la restitution (< 0 = en avance) |
| `previous_ended_rental_id` | Lien avec la location précédente (si consécutive) |
| `time_delta_with_previous_rental_in_minutes` | Délai entre deux locations consécutives |

- **21 310 locations** au total
- Seulement **1 841** locations consécutives (~8,6%) — les cas réellement à risque

### Dataset 2 — Pricing (`get_around_pricing_project.csv`)
- **4 843 véhicules** — 13 features (2 numériques, 4 catégorielles, 7 booléennes)
- **Cible** : `rental_price_per_day` — médiane **119 €/jour**, max 422 €

---

## Slide 3 — EDA : Analyse des Retards

### Findings clés
- **52,1 %** des locations terminées présentent un retard de restitution
- Les voitures **Connect** ont une dispersion de retard légèrement plus élevée
- Seulement **1 841 locations** (~8,6%) ont un prédécesseur direct → seuls ces cas créent de la friction

### Visualisations
- Histogramme des retards par type de checkin (mobile vs Connect)
- Boîte à moustaches des retards — Connect vs Mobile
- Distribution du délai entre locations consécutives

---

## Slide 4 — Simulation du Seuil : le Compromis

### Méthode
Pour chaque seuil `t` (de 0 à 720 min, pas de 30 min) et chaque périmètre (all / connect) :
- **Problèmes résolus** = cas consécutifs où `prev_delay ≤ t`
- **Locations affectées** = consécutives bloquées car `time_delta < t`

### Résultats clés (périmètre : toutes les voitures)

| Seuil | Problèmes résolus | Locations affectées |
|-------|------------------|---------------------|
| 60 min | 46.8% | 1.9% |
| 120 min | 67.4% | 3.1% |
| 180 min | 76.6% | 4.1% |

###  Recommandation
**Seuil de 60 minutes — toutes les voitures**
- Résout **~47% des problèmes** pour seulement **~1.9% des locations affectées**
- Si budget contraint → Connect uniquement : même protection, impact revenu réduit de moitié

---

## Slide 5 — EDA Pricing : Comprendre le Prix

### Corrélations (variables numériques & booléennes)
- **`engine_power`** : corrélation positive la plus forte avec le prix (~0,55)
- **`mileage`** : légère corrélation négative (~-0,12) — normal, voiture usée = moins chère
- **`has_gps`**, **`has_getaround_connect`** : premium de ~+10€/jour chacun

### Prix par catégorie
- **Type de voiture** : convertible > coupé > SUV > berline > citadine
- **Carburant** : électrique & hybride légèrement plus chers
- **Marque** : Porsche, Maserati >> Citroën, Renault

### Insight clé
> La **puissance moteur** et le **type de voiture** sont les prédicteurs les plus forts du prix.

---

## Slide 6 — Feature Engineering & Pipeline

### Prétraitement
```
ColumnTransformer
├── StandardScaler       ← mileage, engine_power
├── OneHotEncoder        ← model_key, fuel, paint_color, car_type
└── passthrough          ← 7 features booléennes (déjà en 0/1)
```

### Pourquoi ces choix ?
- `StandardScaler` évite que les grandes valeurs numériques dominent le modèle
- `OneHotEncoder` encode sans ordre implicite (contrairement à LabelEncoder)
- `Pipeline` Scikit-Learn assure l'absence de data leakage train/test

---

## Slide 7 — Modélisation : Comparaison des Modèles

### 3 modèles comparés
| Modèle | Train R² | Test R² | RMSE | MAE |
|--------|----------|---------|------|-----|
| LinearRegression (baseline) | ~0.56 | ~0.54 | ~22 €| ~16 € |
| **RandomForest (Optimisé)** | **~0.95** | **0.73** | **16.9 €** | **11.1 €** |
| GradientBoosting | ~0.89 | ~0.79 | ~15 € | ~11 € |

### Tracking MLflow
- Toutes les expériences loggées dans `mlflow/mlruns/`
- Paramètres, métriques et modèles enregistrés
- Sélection automatique du meilleur modèle par Test R²

---

## Slide 8 — Résultats & Explicabilité

### Performance du modèle final (RandomForest)
- **Test R² = 0.729** — le modèle explique 73% de la variance du prix
- **RMSE = 16.9 €/day** — erreur quadratique moyenne
- **MAE = 11.1 €/day** — erreur absolue moyenne de ~11 €

### Top 5 Features les plus importantes
1. `engine_power` — ~47%
2. `mileage` — ~27%
3. `has_gps` — ~4%
4. `has_getaround_connect` — ~3%
5. `car_type_suv` — ~2%

> **Insight business** : Le prix est avant tout déterminé par la puissance et le type de véhicule, pas par les équipements.

---

## Slide 9 — Architecture de Production

```
Utilisateur / Équipe Métier
         │
         ▼
   FastAPI (API REST)              Hugging Face Spaces (Docker SDK)
   ┌─────────────────┐
   │  GET  /         │  ← Welcome
   │  GET  /health   │  ← Monitoring
   │  GET  /cars/stats│  ← Business insights
   │  POST /predict  │  ← Prédiction prix (1 voiture)
   │  POST /predict/batch  ← Prédiction prix (n voitures)
   │  GET  /docs     │  ← Documentation Swagger auto
   └────────┬────────┘
            │
            ▼
   RandomForestRegressor
   (model.pkl — joblib)
   
   MLflow Tracking
   (mlflow/mlruns/)
```

---

## Slide 10 — Dashboard Streamlit

**Public cible** : Product Manager / Équipe Produit

**Fonctionnalités :**
- Slider interactif : seuil de 0 à 720 minutes
- Radio : périmètre (toutes voitures / Connect uniquement)
- **KPIs en temps réel** : % problèmes résolus, % revenu à risque
- Graphiques Plotly interactifs : histogrammes, box plots, courbes de trade-off

**Déploiement** : Hugging Face Spaces (SDK Streamlit)
URL : `https://huggingface.co/spaces/philippetos/GetAround`

---

## Slide 11 — Conformité Bloc 5 Jedha

| Critère | Livrable |
|---------|---------|
| Standardiser l'environnement ML | `Dockerfile` + `requirements.txt` |
| Tracking MLflow | Expériences loggées dans `mlflow/mlruns/` |
| API accessible en production | FastAPI déployée sur Hugging Face |
| Interface web intégrée | Dashboard Streamlit déployé |
| Endpoint `/predict` | `POST /predict` + `POST /predict/batch` |
| Documentation `/docs` | Swagger UI auto-généré par FastAPI |

---

## Slide 12 — Conclusion

### Résumé des livrables
1.  **Notebook EDA + ML** (EN + FR) — 46 cellules, 17 graphiques Plotly
2.  **Dashboard Streamlit** — analyse retards interactive
3.  **API FastAPI** — 4 endpoints + Swagger docs
4.  **Docker** — déployable sur Hugging Face Spaces

### Recommandations business
-  **Seuil 60 min sur toutes les voitures** — meilleur compromis satisfaction/revenu
-  **Modèle RF** : erreur de ~10 €/jour — utilisable pour guider les propriétaires
-  Piste d'amélioration : enrichir le dataset avec la localisation (ville vs rural)
