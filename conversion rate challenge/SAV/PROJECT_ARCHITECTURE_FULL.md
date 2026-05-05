# 🏛️ L'ARCHITECTURE DU PROJET : LE SÉNAT & SES MONSTRES

Ce document détaille l'intégralité de la solution technique. L'approche est un **Ensemble Voting** dominé par des méthodes de **Gradient Boosting** (Arbres), avec une touche de **Modèle Linéaire** pour la stabilité.

---

## 1. LE CŒUR DU SYSTÈME : LE SÉNAT (Voting Ensemble)

L'idée centrale est de ne jamais laisser un seul modèle décider. Trois "Sénateurs" votent.

### 👤 Sénateur V1 : "Sniper Elite" (L'Expert Classique)
*   **Famille Algorithmique** : **Ensemble Hybride** (Boosting + Linéaire).
*   **Composition & Nature** :
    1.  **XGBoost (x2)** : *Gradient Boosting Decision Trees*. (Arbres de décision séquentiels qui corrigent les erreurs des précédents).
    2.  **LightGBM** : *Histogram-based Gradient Boosting*. (Variante optimisée et très rapide des arbres de décision).
    3.  **GradientBoosting (Sklearn)** : *Gradient Boosting* classique.
    4.  **LogisticRegression** : *Modèle Linéaire*. (Trace une ligne droite/plan pour séparer les classes).
*   *Pourquoi ce mélange ?* Le modèle linéaire (Logistic) apporte de la stabilité là où les arbres (Boosting) peuvent être trop complexes (overfitting).

### 👤 Sénateur V2 : "Le Mathématicien" (Approche Poisson)
*   **Famille Algorithmique** : **Gradient Boosting Regressor** (Arbres de Régression).
*   **Modèle** : `HistGradientBoostingRegressor` (Sklearn).
*   **Particularité** : Il n'utilise pas une fonction de perte classique (Classification) mais une **Loi de Poisson** (Probabilité d'événements rares).
*   *Nature* : C'est bel et bien une forêt d'arbres, mais entraînée pour prédire une *intensité* et non une *classe*.

### 👤 Sénateur V3 : "L'Ultimate" (L'Ingénieur)
*   **Famille Algorithmique** : **Ensemble de Boosting Pur**.
*   **Composition** :
    *   Mêmes arbres que V1 (XGBoost, LightGBM...).
    *   **Ultimate Mix** : Combinaison d'un *Regressor* (Poisson) et d'un *Classifier* (LogLoss), tous deux basés sur des **Arbres de Gradient Boosting**.
*   *Différence* : Ici, aucune Régression Logistique. C'est du "Heavy Metal" (100% Arbres non-linéaires) pour capturer les interactions complexes.

---

## 2. LA CONSTITUTION (Les Règles du Sénat V9)

*   **Famille** : **Système Expert** (Règles "If-Then" manuelles).
*   Ces règles ne sont PAS du Machine Learning. Ce sont des instructions déterministes codées en dur ("Si Age > 45 et Pays = Chine...").

### 📜 Amendement "Mariage Frères" (Les Anciens)
*   Force la décision positive pour les profils âgés et riches.

### 📜 Amendement "Erasmus" (Les Jeunes Européens)
*   Force la décision positive pour les étudiants actifs (Allemagne/UK).

---

## 3. LES OUTSIDERS (Les Armes Secrètes)

### 🔫 FN_SNIPER (L'Hybride Heuristique)
*   **Famille** : **Hybride** (Gradient Boosting + Heuristique).
*   **Composant 1 (85%)** : `HistGradientBoostingClassifier` (Arbres).
*   **Composant 2 (15%)** : Formule mathématique manuelle (`Score = 0.4*Pages + 0.2*Interaction...`).
*   *Nature* : C'est un modèle d'arbres que l'on a "dopé" manuellement pour qu'il soit plus agressif.

### 🦖 LE MONSTRE (Pseudo-Labeling / Last Chance)
*   **Famille** : **Gradient Boosting Classifier** (Arbres).
*   **Modèle** : `HistGradientBoostingClassifier` (Sklearn).
*   **Particularité** : Apprentissage Semi-Supervisé.
*   *Nature* : C'est un pur modèle d'arbres, mais entraîné sur un dataset "augmenté" par les prédictions d'un autre modèle. Il n'y a pas de régression logistique ni de Random Forest ici.

---

## RÉSUMÉ CLASSIFICATION

| Composant | Nature Mathématique | Rôle |
| :--- | :--- | :--- |
| **XGBoost / LightGBM** | **Arbres (Gradient Boosting)** | La puissance brute (non-linéaire). |
| **Logistic Regression** | **Linéaire** | Le garde-fou (stabilité). Utilisé uniquement dans V1. |
| **Poisson Regressor** | **Arbres (Régression)** | La vision "statistique" (événements rares). |
| **Amendements** | **Règles (Logique)** | Le bon sens humain (correction d'erreurs). |
