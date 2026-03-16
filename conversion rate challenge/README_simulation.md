# 🎯 Simulation Gradient Boosting - Conversion Rate Challenge

## 📋 Contexte du projet

Ce projet vise à **prédire si un visiteur d'un site web va se convertir** (effectuer un achat, s'inscrire, etc.) en utilisant un modèle de **Gradient Boosting Classifier**.

---

## 📊 Description des données

### Fichiers disponibles
| Fichier | Description | Taille |
|---------|-------------|--------|
| `conversion_data_train.csv` | Données d'entraînement | 284 580 lignes |
| `conversion_data_test.csv` | Données de test | 31 620 lignes |

### Variables explicatives (Features)

| Variable | Type | Description |
|----------|------|-------------|
| `country` | Catégorielle | Pays d'origine du visiteur (China, Germany, UK, US) |
| `age` | Numérique | Âge du visiteur |
| `new_user` | Binaire | Nouveau visiteur (1) ou visiteur récurrent (0) |
| `source` | Catégorielle | Source du trafic (Ads, Direct, Seo) |
| `total_pages_visited` | Numérique | Nombre de pages visitées durant la session |

### Variable cible

| Variable | Type | Description |
|----------|------|-------------|
| `converted` | Binaire | 0 = Non converti, 1 = Converti |

---

## 📈 Analyse exploratoire

### Distribution de la variable cible

```
converted
    0    275 400  (96.77%)
    1      9 180  ( 3.23%)
```

> ⚠️ **Déséquilibre des classes** : Seulement **3.23%** des visiteurs se convertissent, ce qui représente un problème de classification déséquilibrée.

### Encodage des variables catégorielles

| Variable | Valeurs encodées |
|----------|------------------|
| `country` | China=0, Germany=1, UK=2, US=3 |
| `source` | Ads=0, Direct=1, Seo=2 |

---

## 🔧 Méthodologie

### 1. Préparation des données
- Encodage des variables catégorielles avec `LabelEncoder`
- Division train/validation : **80% / 20%** (stratifié)
  - Entraînement : 227 664 échantillons
  - Validation : 56 916 échantillons

### 2. Configuration du modèle

```python
GradientBoostingClassifier(
    n_estimators=100,      # Nombre d'arbres
    learning_rate=0.1,     # Taux d'apprentissage
    max_depth=5,           # Profondeur maximale des arbres
    min_samples_split=10,  # Échantillons min. pour diviser un nœud
    min_samples_leaf=5,    # Échantillons min. dans une feuille
    random_state=42        # Reproductibilité
)
```

---

## 📊 Résultats

### Métriques de performance

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Accuracy** | 98.62% | Proportion globale de prédictions correctes |
| **F1-Score** | 0.7646 | Moyenne harmonique de la précision et du rappel |
| **ROC-AUC** | 0.9863 | Capacité du modèle à discriminer les classes |

### Matrice de confusion

```
                    Prédit: Non converti    Prédit: Converti
Réel: Non converti       54 859 (TN)            221 (FP)
Réel: Converti              563 (FN)          1 273 (TP)
```

| Terme | Signification | Valeur |
|-------|---------------|--------|
| **TN** (True Negative) | Non convertis correctement prédits | 54 859 |
| **FP** (False Positive) | Faussement prédits comme convertis | 221 |
| **FN** (False Negative) | Convertis ratés par le modèle | 563 |
| **TP** (True Positive) | Convertis correctement prédits | 1 273 |

### Rapport de classification détaillé

| Classe | Précision | Rappel | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Non converti (0) | 0.99 | 1.00 | 0.99 | 55 080 |
| Converti (1) | 0.85 | 0.69 | 0.76 | 1 836 |
| **Moyenne pondérée** | 0.99 | 0.99 | 0.99 | 56 916 |

---

## 🔍 Importance des features

| Rang | Feature | Importance | Visualisation |
|------|---------|------------|---------------|
| 1 | `total_pages_visited` | **84.77%** | ██████████████████████████████████████████ |
| 2 | `country_encoded` | 6.34% | ███ |
| 3 | `new_user` | 5.68% | ██ |
| 4 | `age` | 2.97% | █ |
| 5 | `source_encoded` | 0.23% | ▪ |

### Interprétation

1. **`total_pages_visited` (84.77%)** : C'est de loin le facteur le plus déterminant. Plus un visiteur consulte de pages, plus il est susceptible de se convertir.

2. **`country_encoded` (6.34%)** : Le pays d'origine a une influence modérée. L'Allemagne semble avoir un meilleur taux de conversion.

3. **`new_user` (5.68%)** : Les utilisateurs récurrents convertissent mieux que les nouveaux.

4. **`age` (2.97%)** : L'âge a un impact mineur sur la conversion.

5. **`source_encoded` (0.23%)** : La source du trafic (Ads, Direct, SEO) n'influence presque pas la conversion.

---

## 🎯 Prédictions sur les données de test

| Métrique | Valeur |
|----------|--------|
| Nombre total de visiteurs | 31 620 |
| Prédictions de conversion | 842 |
| Taux de conversion prédit | 2.66% |

Les prédictions sont sauvegardées dans `predictions_gradient_boosting.csv`.

---

## 📁 Fichiers du projet

| Fichier | Description |
|---------|-------------|
| `conversion_data_train.csv` | Données d'entraînement originales |
| `conversion_data_test.csv` | Données de test originales |
| `gradient_boosting_simulation.py` | Script Python de la simulation |
| `predictions_gradient_boosting.csv` | Prédictions générées |
| `README_simulation.md` | Ce document |

---

## 💡 Conclusions et recommandations

### Points clés
- ✅ Le modèle Gradient Boosting atteint une excellente performance globale (**98.62% accuracy**)
- ✅ Le **F1-Score de 0.76** est bon pour une classe minoritaire à 3.23%
- ✅ Le **ROC-AUC de 0.99** indique une très bonne capacité de discrimination

### Pour améliorer la détection des conversions
1. **Optimiser le seuil de décision** : Baisser le seuil de 0.5 pour augmenter le rappel
2. **Rééquilibrer les classes** : Utiliser SMOTE ou le paramètre `class_weight`
3. **Feature engineering** : Créer des features combinées (ex: pages_par_minute)
4. **Hyperparameter tuning** : GridSearchCV pour optimiser les paramètres

---

*Document généré le 15 décembre 2025*
