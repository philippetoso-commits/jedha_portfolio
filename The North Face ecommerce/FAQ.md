# FAQ – The North Face E-commerce
## Questions Techniques Anticipées du Jury
---

### Question 1 : Pourquoi avoir choisi DBSCAN plutôt que K-Means pour le clustering ?
**Réponse attendue :**

K-Means présente **deux problèmes majeurs pour du texte** :

1. **La distance euclidienne est inadaptée** aux vecteurs TF-IDF creux et de haute dimension. La distance euclidienne est sensible à la norme des vecteurs (longueur du document), ce qui introduit un biais : deux descriptions identiques sur le fond mais l'une plus longue que l'autre seront jugées très différentes.

2. **K-Means nécessite de spécifier k à l'avance.** Or, nous ne savons pas combien de catégories de produits existent réellement dans les données — c'est justement ce que le modèle doit découvrir.

**DBSCAN résout ces deux problèmes :**
- Il utilise `metric='cosine'`, qui mesure l'angle entre deux vecteurs TF-IDF, indépendamment de leur norme → il compare les thématiques, pas les longueurs
- Il découvre automatiquement le nombre de clusters en fonction de la densité locale des points
- Il identifie les **outliers** (cluster = -1) plutôt que de les forcer dans un groupe — ce qui est utile pour détecter des produits atypiques dans le catalogue

**Limites de DBSCAN que j'accepte :**
- Sensible aux hyperparamètres `eps` et `min_samples` → j'ai justifié ma sélection via un sweep systématique et le Silhouette Score

---

### Question 2 : Qu'est-ce que l'overfitting et est-ce un risque ici ?
**Réponse attendue :**

L'**overfitting** (surapprentissage) se produit quand un modèle apprend trop précisément les données d'entraînement — y compris le bruit — au point de mal généraliser sur de nouvelles données.

**Dans le cadre d'un projet supervisé**, l'overfitting se détecte par un fort écart entre les performances sur les données d'entraînement et de test (ex. : accuracy = 99% sur train / 60% sur test).

**Dans notre projet non supervisé**, la notion d'overfitting est différente :
- Il n'y a pas de variable cible → pas de généralisation à proprement parler
- Le "risque" équivalent est une **solution trop spécifique** : des paramètres TF-IDF ou DBSCAN qui créent des clusters parfaitement adaptés à ces 500 produits, mais qui s'effondreraient si on ajoutait 500 nouveaux produits

**Mesures préventives appliquées :**
- **min_df=2** dans TF-IDF : ignore les termes n'apparaissant qu'une seule fois (bruit)
- **max_df=0.85** : ignore les termes trop communs qui n'apportent pas d'information différentiante
- **min_samples=2** dans DBSCAN : évite les singletons (groupes d'un seul produit)

---

### Question 3 : Pourquoi avoir utilisé TF-IDF plutôt que des embeddings (Word2Vec, BERT) ?
**Réponse attendue :**

C'est une excellente question qui permet de montrer la connaissance du spectre des approches NLP.

**TF-IDF** est une représentation **bag-of-words** (sac de mots) — elle ignore l'ordre des mots et leur contexte sémantique profond. Ses avantages pour ce projet :
- Simple, rapide, interprétable (on peut lire les termes importants)
- Fonctionne bien avec peu de données (~500 documents)
- Recommandé explicitement dans le sujet du projet (avec `TFIDFVectorizer`)
- Compatible directement avec DBSCAN et TruncatedSVD sans transformation supplémentaire

**Les embeddings (BERT, Sentence-Transformers)** offriraient :
- Une meilleure capture du contexte sémantique (polysémie, synonymes)
- Des représentations plus denses et moins bruitées

Mais avec des **inconvénients** dans ce contexte :
- Beaucoup plus lents à calculer
- Moins interprétables (pas de termes lisibles pour les WordClouds)
- Pas directement utilisables avec TruncatedSVD pour LSA

**En production**, avec plus de ressources, je migrerais vers des Sentence-Transformers (ex. `all-MiniLM-L6-v2`) couplés à HDBSCAN pour de meilleures performances.

---

### Question 4 : Comment évaluer la performance d'un modèle non supervisé ?
**Réponse attendue :**

C'est l'un des grands défis du Machine Learning non supervisé : **il n'y a pas de "bonne réponse"** à comparer.

**Métriques quantitatives utilisées :**

1. **Silhouette Score** (pour le clustering) :
   - Mesure à quel point un point est similaire à son propre cluster vs les autres clusters
   - Valeur entre -1 et +1 : proche de 1 = clusters denses et bien séparés
   - Formule : `(b - a) / max(a, b)` où `a` = distance intra-cluster, `b` = distance inter-cluster
   - Ici utilisé avec `metric='cosine'` pour être cohérent avec DBSCAN

2. **Variance Expliquée** (pour LSA/TruncatedSVD) :
   - Proportion de la variance totale de la matrice TF-IDF capturée par les topics
   - Analogue au R² en régression

**Évaluation qualitative** :
- **WordClouds** : si les mots d'un cluster sont sémantiquement cohérents, le cluster est pertinent
- **Exemples manuels** : vérifier que les 5 recommandations d'un produit connu sont logiques

**Limites :**
- Le Silhouette Score est biaisé vers les clusters sphériques — moins adapté à DBSCAN
- L'évaluation humaine reste indispensable en NLP non supervisé

---

### Question 5 : Expliquez ce qu'est le Data Leakage et comment vous l'avez évité
**Réponse attendue :**

Le **Data Leakage** (fuite de données) se produit lorsque des informations du futur ou du jeu de test "contaminent" l'entraînement du modèle, produisant des résultats artificiellement bons qui ne se retrouveront pas en production.

**Exemple concret de leakage :**
```python
# MAUVAIS : TF-IDF ajusté sur tout le corpus AVANT le split
tfidf = TfidfVectorizer().fit(df['description'])
X_train_tfidf = tfidf.transform(X_train)  # Le vocabulaire inclut des mots du test !
X_test_tfidf  = tfidf.transform(X_test)
```

**Dans notre projet non supervisé :**
- Il n'y a **pas de split train/test** (apprentissage non supervisé sur tout le corpus)
- Le risque de leakage est donc minimal par nature
- **Cependant**, nous utilisons un `Pipeline` Scikit-Learn pour garantir que le TF-IDF est toujours ajusté (`fit`) UNIQUEMENT sur les données d'entrée, jamais sur un sous-ensemble que le modèle n'est pas censé voir

```python
# CORRECT avec Pipeline : le preprocesseur est encapsulé
tfidf_pipeline = Pipeline([
    ('preprocesseur', FunctionTransformer(preprocess_text, validate=False)),
    ('tfidf', TfidfVectorizer(max_features=3000))
])
tfidf_matrix = tfidf_pipeline.fit_transform(df['description'])
```

Ce pattern `Pipeline` est **LA bonne pratique** pour garantir la reproductibilité et l'absence de leakage quand on passe en production (le pipeline peut être sérialisé avec `pickle`/`joblib` et appliqué à de nouvelles descriptions).

---

### Question 6 (Bonus) : Quelle est la différence entre clustering et topic modeling, et pourquoi faire les deux ?
**Réponse attendue :**

**Clustering (DBSCAN) :**
- Assigne **un seul cluster** (catégorie) à chaque produit
- Vision **exclusive** : un produit = un groupe
- Utile pour le recommender system (comparer des produits de même catégorie)

**Topic Modeling (LSA) :**
- Assigne **un mélange de sujets** à chaque produit
- Vision **additive** : un produit peut toucher plusieurs thèmes (ex. une veste de randonnée est à la fois `outdoor`, `waterproof`, et `lightweight`)
- Utile pour comprendre la sémantique profonde et améliorer la structure du catalogue

**Complémentarité :**
- Le clustering donne une **taxonomie opérationnelle** du catalogue (catégories claires pour filtrer)
- Le topic modeling révèle une **représentation sémantique latente** (insights pour le SEO, les tags, la navigation)
- Les deux approches ensemble donnent une vision complète et complémentaire du corpus

---

*Document préparé pour la soutenance du projet The North Face E-commerce — Jedha Bootcamp*
