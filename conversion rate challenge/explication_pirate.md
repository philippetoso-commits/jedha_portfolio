# Explication détaillée du script "Pirate" (pirate.ipynb)

Ce fichier représente une stratégie avancée, souvent utilisée en compétition (Kaggle) pour gagner les derniers centièmes de précision nécessaires pour passer un palier (ici le fameux "0.77").

Il combine trois techniques puissantes : **L'Ensemble Learning**, **Le Feature Engineering Polynomial**, et surtout le **Pseudo-Labeling** (d'où le nom "Pirate", car on "vole" de l'information au jeu de test).

---

## 1. Feature Engineering & Preprocessing

### Création de variables manuelles
Avant tout traitement automatique, le script crée des interactions explicites qui ont du sens métier :
*   **`interaction_age_pages`** : `age * total_pages_visited`. Cette variable capture le fait que l'impact du nombre de pages visitées peut varier selon l'âge.
*   **`is_active`** : Binaire. Indique si l'utilisateur a vu plus de 2 pages. C'est un indicateur fort d'intérêt minimum.
*   **`pages_per_age`** : Ratio. Tentative de normaliser l'activité par l'âge.

### Feature Engineering Polynomial (Le "Mur" de complexité)
C'est une étape clé pour capturer les non-linéarités manquantes.
*   **Méthode** : `PolynomialFeatures(degree=2)`
*   **Ce que ça fait** : Pour chaque variable numérique (A, B...), cela crée automatiquement :
    *   Les carrés : $A^2, B^2$
    *   Les interactions croisées : $A \times B$
*   **Pourquoi ?** Certains modèles (comme la régression logistique, mais aussi parfois les arbres peu profonds) peinent à modéliser des courbes complexes (comme une parabole). Lui donner $X^2$ directement l'aide énormément.

---

## 2. Le Modèle : Un "Voting Classifier" Robuste

Au lieu de miser sur un seul cheval, le script utilise une "équipe" de modèles (Ensemble Learning) pour stabiliser les prédictions et réduire la variance.

### Les Composants
1.  **XGBoost** (`clf_xgb`) : Le standard de l'industrie. Rapide et précis.
2.  **LightGBM** (`clf_lgbm`) : Souvent légèrement différent du XGBoost dans sa manière de créer les arbres (leaf-wise vs depth-wise), ce qui apporte de la diversité.
3.  **GradientBoosting** (`clf_gb`) : L'implémentation standard de Scikit-Learn. Plus lente, mais complète bien les deux autres.

### Les Hyperparamètres
Ils sont configurés en mode "Douceur" (pour éviter le sur-apprentissage) :
*   `n_estimators=300` : Nombre d'arbres modéré.
*   `max_depth=4` : Arbres peu profonds (limite la complexité).
*   `learning_rate=0.05` : Apprentissage lent (nécessite plus d'arbres mais généralise mieux).
*   `subsample=0.9` : Chaque arbre ne voit que 90% des données (réduit la variance).

### Le Vote (`VotingClassifier`)
Les trois modèles votent ensemble selon la méthode **'soft' voting**.
*   Ils ne votent pas juste "0" ou "1", mais donnent leur probabilité (ex: 0.82).
*   La moyenne pondérée de ces probabilités est calculée.
*   **Poids** : `[1.2, 1.2, 0.8]` -> On fait un peu plus confiance à XGBoost et LightGBM qu'au GradientBoosting standard.

---

## 3. La Technique "Pirate" : Le Pseudo-Labeling

C'est le cœur de la stratégie pour dépasser le score plafond. L'idée est d'utiliser les données du Test (dont on n'a pas la réponse) pour ré-entraîner le modèle.

**Comment ça marche ? Le processus étape par étape :**

1.  **Entraînement Initial** : On entraîne le modèle `VotingClassifier` normalement sur le jeu de Train (`X`, `y`).
2.  **Prédiction sur le Test** : Le modèle fait ses prédictions (probabilités) sur le jeu de Test (`X_test`).
3.  **Filtrage par Confiance (Le tri)** :
    *   On regarde où le modèle est **extrêmement sûr** de lui.
    *   Cas Positifs sûrs : Probabilité > **90%** (0.90).
    *   Cas Négatifs sûrs : Probabilité < **5%** (0.05).
    *   *Note : On rejette tout ce qui est entre 0.05 et 0.90 (zone d'incertitude).*
4.  **Création des "Faux Labels" (Pseudo-Labels)** :
    *   On prend ces lignes du Test "sûres".
    *   On leur **colle l'étiquette devinée** par le modèle comme si c'était la vérité absolue (1 pour les >0.90, 0 pour les <0.05).
5.  **Augmentation du Train** :
    *   On **ajoute** ces lignes du Test (avec leurs faux labels) au jeu d'Entraînement original.
    *   Résultat : Un jeu d'entraînement plus gros, qui contient maintenant des "exemples" issus de la distribution du Test.
6.  **Re-Entraînement Final** :
    *   On oublie le premier modèle.
    *   On **ré-entraîne** un modèle tout neuf sur ce nouveau gros jeu de données "Augmenté".

**Pourquoi ça marche ?**
*   **Adaptation au domaine** : Si la distribution du Test est légèrement différente du Train (Data Drift), le modèle apprend cette nouvelle distribution grâce aux pseudo-labels.
*   **Densification de la frontière de décision** : En ajoutant des points "sûrs", on force le modèle à être plus robuste dans les zones claires, ce qui peut l'aider à mieux trancher les cas limites proches.

---

## 4. Prédiction Finale

Le modèle final (re-entraîné) est utilisé pour prédire une dernière fois sur le Test. Un seuil de décision (`0.42`) est appliqué pour convertir les probabilités en classes (0 ou 1).

*Note sur le seuil : 0.42 est souvent optimal pour le F1-Score sur des datasets déséquilibrés où l'on veut récupérer un peu plus de classe 1 sans trop perdre en précision.*
