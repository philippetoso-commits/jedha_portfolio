# GetAround — FAQ Jury
## Jedha Bootcamp — Bloc 5

---

### Q1 — Pourquoi avoir choisi Random Forest plutôt qu'un XGBoost ou un réseau de neurones ?

**Réponse :**

Le choix du Random Forest repose sur trois critères principaux :

1. **Performance** : sur ce dataset tabulaire de taille modeste (4 843 lignes, 13 features), les forêts aléatoires offrent de bons résultats — Test R² ≈ 0,73 — sans nécessiter de tuning complexe.

2. **Interprétabilité** : le Random Forest fournit nativement une **Feature Importance** basée sur la réduction d'impureté (Gini), ce qui permet d'expliquer les prédictions aux équipes métier. C'est un critère essentiel dans un contexte de pricing.

3. **Robustesse** : contrairement au Gradient Boosting ou XGBoost, le Random Forest est naturellement moins sensible aux hyperparamètres et aux outliers grâce au bagging.

J'ai quand même testé un Gradient Boosting en comparaison — les performances sont très proches. J'aurais pu tester XGBoost, mais le gain marginal probable ne justifiait pas la complexité ajoutée pour ce cas d'usage.

Un réseau de neurones aurait été inadapté ici : les données sont tabulaires et peu volumineuses — le deep learning brille sur des données non structurées (images, texte) ou à très grande échelle.

---

### Q2 — Qu'est-ce qu'un overfitting ? Comment avez-vous évité ce problème ?

**Réponse :**

L'overfitting (sur-apprentissage) survient quand un modèle **mémorise** les données d'entraînement au lieu d'en extraire des **patterns généralisables**. Concrètement : le modèle est très performant sur le train set mais chute sur des données nouvelles.

On le détecte en comparant les métriques Train vs Test : un Train R² très élevé avec un Test R² bien inférieur est le signal classique.

Dans ce projet, j'ai mis en place plusieurs garde-fous :

- **Séparation train/test** à 80/20 avant tout prétraitement — aucune information du test set ne pollue le pipeline.
- **Pipeline Scikit-Learn** complet : le scaler et l'encoder sont `fit` uniquement sur le train set puis `transform` sur le test — pas de data leakage.
- **GridSearchCV avec cross-validation à 5 folds** : l'hyperparamètre tuning est réalisé sur le train set seulement, avec validation croisée pour éviter d'optimiser sur du bruit.
- **Paramètre `min_samples_split`** : contrôle la profondeur des arbres pour éviter des arbres trop spécialisés.

Le modèle final affiche Train R² ≈ 0,95 vs Test R² ≈ 0,73 — il y a un overfit modéré, ce qui est normal pour un Random Forest, mais les performances test restent exploitables.

---

### Q3 — Pourquoi avoir utilisé OneHotEncoder et pas LabelEncoder pour les variables catégorielles ?

**Réponse :**

Le `LabelEncoder` transforme les catégories en entiers : `diesel → 0`, `petrol → 1`, `hybrid → 2`, `electro → 3`. Ce faisant, il **introduit implicitement un ordre numérique** entre les catégories — comme si l'électrique était "trois fois plus" que le diesel, ce qui n'a aucun sens.

Le `OneHotEncoder` crée une **colonne binaire par modalité** — sans ordre implicite. Chaque catégorie est indépendante. C'est le choix correct pour des variables nominales comme `fuel`, `car_type`, `model_key`.

Le seul inconvénient est l'augmentation de la dimensionnalité (par exemple `model_key` avec 10 marques génère 10 colonnes). Mais dans le cas présent, le nombre total de features après encodage reste gérable pour un Random Forest, qui gère nativement les hautes dimensions grâce à la sélection aléatoire de features à chaque nœud.

---

### Q4 — Comment fonctionne votre API et pourquoi FastAPI plutôt que Flask ?

**Réponse :**

L'API expose **4 endpoints** :
- `GET /health` — monitoring opérationnel
- `GET /cars/stats` — statistiques du dataset de référence
- `POST /predict` — prédiction pour un véhicule (champs nommés en entrée et sortie)
- `POST /predict/batch` — prédictions en lot pour plusieurs voitures

FastAPI offre plusieurs avantages vs Flask pour ce cas d'usage :

1. **Documentation automatique** : FastAPI génère un **Swagger UI** natif à `/docs` et un ReDoc à `/redoc` — sans configuration. C'est d'ailleurs une exigence du projet.
2. **Validation des données intégrée** : grâce à Pydantic, les types des inputs sont vérifiés automatiquement — pas besoin de gérer manuellement les erreurs de format.
3. **Performance** : basé sur ASGI (Starlette), FastAPI est significativement plus rapide que Flask (WSGI) pour des requêtes concurrentes.
4. **Typage et auto-complétion** : facilite la maintenance et la documentation du code.

Flask reste pertinent pour des projets simples ou quand on hérite d'une codebase existante. FastAPI est le standard moderne pour des APIs ML en production.

---

### Q5 — Comment avez-vous déterminé le seuil de 60 minutes ? N'est-ce pas arbitraire ?

**Réponse :**

C'est une excellente question — justement, ce n'est pas arbitraire. La recommandation de 60 minutes est issue d'une **simulation quantitative**.

Pour chaque seuil de 0 à 720 minutes (par pas de 30 min), j'ai calculé deux métriques :
- Le **% de cas problématiques résolus** : parmi les locations consécutives où le retard cause une friction, combien seraient évitées par ce seuil ?
- Le **% de locations affectées** : parmi toutes les locations, combien seraient bloquées (revenu perdu) ?

La courbe de trade-off montre un **coude à 60 minutes** : au-delà, les gains marginaux en résolution de problèmes deviennent faibles alors que les coûts en revenu augmentent rapidement.

À 60 min :
- ≈ 47% des problèmes résolus
- ≈ 1.9% du revenu affecté

À 120 min :
- ≈ 67% des problèmes résolus
- ≈ 3.1% du revenu affecté

Le rapport bénéfice/coût est clairement meilleur à 60 minutes. Mais le choix final appartient au Product Manager, qui peut avoir des contraintes additionnelles — c'est pourquoi j'ai construit le **dashboard interactif** : il lui permet d'explorer lui-même ces courbes en temps réel avec le périmètre de son choix.

---

### Q6 — Qu'est-ce que MLflow et pourquoi l'avez-vous utilisé ?

**Réponse :**

MLflow est un outil open-source de **gestion du cycle de vie des expériences ML** (ML Lifecycle Management). Il permet de :

- **Logger les hyperparamètres** de chaque run (n_estimators, max_depth, etc.)
- **Logger les métriques** (R², RMSE, MAE) pour chaque expérience
- **Sauvegarder les artefacts** — le modèle sérialisé, les graphiques de performance
- **Comparer visuellement** plusieurs runs côte à côte

Dans ce projet, j'ai tracké les **3 modèles** (LinearRegression, RandomForest, GradientBoosting) dans une expérience nommée `getaround_pricing`. MLflow enregistre automatiquement les runs dans `mlflow/mlruns/`.

L'intérêt pratique : si je retravaille le modèle dans 6 mois — avec de nouvelles données ou de nouveaux hyperparamètres — j'ai une **traçabilité complète** de l'historique. Je peux comparer les nouvelles performances avec les anciennes en une commande : `mlflow ui`.

C'est une bonne pratique indispensable en contexte professionnel, et une exigence explicite du Bloc 5 Jedha.
