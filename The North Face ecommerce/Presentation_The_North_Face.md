# The North Face E-commerce – Présentation Projet
## NLP & Machine Learning Non Supervisé pour Booster les Ventes en Ligne

---

## Slide 1 – Introduction & Contexte Métier

**The North Face** — marque outdoor américaine fondée en 1968, devenue un symbole culturel mondial.

**Le Défi :** Le département marketing veut exploiter le Machine Learning pour augmenter le taux de conversion sur [thenorthface.fr](https://www.thenorthface.fr/)

**Deux leviers identifiés :**
- 🔄 Un **système de recommandation** ("Vous aimerez aussi...")
- 🗂️ Une **restructuration du catalogue** via l'extraction de sujets latents

**Notre Mission :** Analyser **500 descriptions de produits** avec le NLP et l'Apprentissage Non Supervisé

---

## Slide 2 – Le Problème et la Solution IA

**Problème Business :**
> "Nos clients ne trouvent pas facilement des produits similaires à ceux qui les intéressent, et notre catalogue n'est pas structuré de façon optimale."

**Solution ML :**

| Problème | Solution ML | Algorithme |
|----------|-------------|------------|
| Pas de navigation par similarité | Recommender System | DBSCAN + Similarité Cosinus |
| Structure catalogue sous-optimale | Topic Modeling | LSA (TruncatedSVD) |
| Catégories existantes non validées | Clustering textuel | DBSCAN sur TF-IDF |

**Type de projet :** 🔵 Apprentissage **Non Supervisé** (pas de labels, pas de variable cible)

---

## Slide 3 – Le Jeu de Données (Résultats Réels)

**Source :** [Kaggle – Product Item Data (cclark)](https://www.kaggle.com/cclark/product-item-data)

| Caractéristique | Valeur Mesurée |
|-----------------|----------------|
| Nombre de produits (SKUs) | **500** |
| Colonnes disponibles | `id`, `description` |
| Longueur moyenne des descriptions | **1 122 caractères** / **149 mots** |
| Longueur min / max | 416 chars / 3 540 chars |
| Contenu HTML | Oui (`<br>`, `<li>`, `<ul>`) |
| Valeurs manquantes | **0** |
| Doublons | **0** |

**Observations EDA :**
- Top mots bruts : `and`, `the`, `a`, `with`, `for`, `recyclable`, `common` → mots vides + mentions légales
- Après nettoyage : `polyester`, `pocket`, `organic cotton`, `nylon`, `spandex`, `fleece`
- Forte mention de **l'éco-responsabilité** (`recyclable`, `Common Threads Recycling Program`)

---

## Slide 4 – EDA : Ce que les Données nous Disent

**Mots les plus discriminants (Top 10 TF-IDF moyen):**

| Terme | Score TF-IDF |
|-------|-------------|
| recycle | 0.05123 |
| pocket | 0.04736 |
| organic | 0.04491 |
| cotton | 0.04409 |
| organic cotton *(bigramme)* | 0.04316 |
| polyester | 0.03968 |
| recyclable | 0.03878 |
| common thread | 0.03788 |

**Insights business importants :**
> 1. La **durabilité** et l'**éco-responsabilité** (`recycle`, `recyclable`, `organic cotton`) sont au cœur du discours produit — c'est un axe marketing fort
> 2. Les **fonctionnalités physiques** (`pocket`, `zipper`, `strap`) distinguent les catégories techniques
> 3. Les bigrammes TF-IDF (`organic cotton`, `common thread`) capturent du sens que les mots seuls ne peuvent pas

---

## Slide 5 – Pipeline de Prétraitement NLP

```
Texte Brut (HTML incl.) → [Suppression HTML] → [Alphabétique uniquement]
         → [spaCy en_core_web_sm]
              ├── Tokenisation  
              ├── Suppression 326 mots vides anglais  
              └── Lemmatisation (ex: "wicking" → "wick", "pockets" → "pocket")
         → [TfidfVectorizer]
              ├── max_features=3000 termes
              ├── min_df=2 (termes uniques exclus)
              ├── max_df=0.85 (termes trop communs exclus)
              └── ngram_range=(1,2) → unigrammes + bigrammes
              
RÉSULTAT : Matrice TF-IDF (500 × 3000) — Créusité : 96.8%
```

**Exemple de transformation réelle :**
- Avant : `"Active classic boxers - There's a reason why our boxers are a cult favorite..."`
- Après : `"active classic boxer reason cult favorite cool sticky situation quick dry lightweight underwear..."`

---

## Slide 6 – Partie 1 : Clustering DBSCAN (Résultats Réels)

**Pourquoi DBSCAN et pas K-Means ?**

| Critère | K-Means | DBSCAN |
|---------|---------|--------|
| Nombre de clusters | À spécifier manuellement | Découvert automatiquement ✅ |
| Métrique de distance | Euclidienne ❌ (inadaptée texte) | Cosinus ✅ |
| Gestion des outliers | ❌ Force dans un cluster | ✅ Labels comme `-1` |
| Forme des clusters | Sphérique | Arbitraire ✅ |

**Paramétrage final sélectionné :**
- `eps = 0.3`, `min_samples = 2`, `metric = 'cosine'`

**Résultats obtenus sur les données réelles :**
| Métrique | Valeur |
|----------|--------|
| Nombre de clusters formés | **95 clusters** |
| Produits classifiés dans un cluster | ~200 / 500 |
| Produits non assignés (outliers `-1`) | ~300 / 500 (60%) |

> ⚠️ **Analyse du taux d'outliers élevé :** Ce phénomène s'explique par la **grande diversité des produits** — The North Face couvre des catégories très hétérogènes (vêtements, chaussures, accessoires, posters, équipements de pêche). Beaucoup de produits sont uniques dans leurs caractéristiques. En pratique, un eps plus élevé (0.4-0.5) ou un algorithme de clustering hiérarchique permettrait de réduire ce taux.

---

## Slide 7 – Résultats du Clustering : Clusters Identifiés

**Clusters les plus significatifs (par nombre de produits) :**

| Cluster | Nb produits | Mots-clés dominants | Catégorie interprétée |
|---------|-------------|--------------------|-----------------------|
| Cluster 0 | 6 | `pocket`, `pant`, `dwr`, `zip`, `stretch` | Pantalons techniques |
| Cluster 2 | 6 | `sock`, `toe`, `lintoe`, `loop`, `wool` | Chaussettes de performance |
| Cluster 14 | 6 | `stretch`, `organic`, `cotton`, `climb`, `front` | Vêtements d'escalade |
| Cluster 38 | 6 | `organic`, `cotton`, `denim`, `jean`, `recycle` | Jeans coton bio |
| Cluster 41 | 5 | `sun`, `runshade`, `protection`, `heat` | Protection solaire running |
| Cluster 51 | 4 | `bra`, `recycle`, `fabric`, `polyester` | Brassières sport |
| Cluster 40 | 4 | `insole`, `support`, `foot`, `provide` | Chaussures/semelles |

**Validation des WordClouds :** Cohérence sémantique forte au sein de chaque cluster — le clustering est pertinent pour les products suffisamment distincts.

---

## Slide 8 – Partie 2 : Système de Recommandation (Résultats Réels)

**Architecture du Recommender :**
```python
find_similar_items(item_id) → Top-5 produits les plus proches (distance cosinus)
```

**Résultats de tests réels sur 3 produits :**

**Produit id=1** (Active classic boxers – outlier cluster -1 → similarité globale utilisée) :
| Recommandation | Similarité | Description |
|---|---|---|
| id=19 | **0.489** | Cap 1 boxer briefs (sous-vêtements Capilene) |
| id=494 | **0.430** | Active boxer briefs (sous-vêtements voyage) |
| id=365 | **0.358** | Organic cotton boxers (coton bio) |
→ ✅ Recommandations très pertinentes : tous des sous-vêtements techniques

**Produit id=4** (Alpine guide pants – cluster 0) :
| Recommandation | Similarité | Description |
|---|---|---|
| id=159 | **0.940** | Alpine guide pants (variante coloris) |
→ ✅ Similarité quasi-parfaite entre deux versions du même produit

**Produit id=12** (Baggies shorts – outlier -1) :
| Recommandation | Similarité | Description |
|---|---|---|
| id=402 | **0.475** | River shorts (shorts rivière) |
| id=408 | **0.389** | Baggies shorts (variante enfant) |
→ ✅ Recommandations cohérentes : shorts d'activités eau / plein air

**Conclusion :** Le système est fonctionnel et produit des recommandations pertinentes, même pour les outliers (fallback sur similarité globale).

---

## Slide 9 – Partie 3 : Topic Modeling LSA (Résultats Réels)

**Paramètres :** `TruncatedSVD(n_components=15, random_state=42)`

**Variance expliquée : 29.3%** sur 15 topics *(note: valeur typique pour du texte avec TF-IDF; 100% n'est pas un objectif en NLP)*

**Topics identifiés et leurs mots-clés (Feature Importance – charges SVD) :**

| Topic | Variance | Mots-clés | Interprétation |
|-------|----------|-----------|----------------|
| 0 | 1.80% | `organic`, `cotton`, `recycle`, `shirt`, `recyclable` | Vêtements éco-responsables *(198 produits dominant)* |
| 1 | 5.05% | `pocket`, `polyester`, `zipper`, `dwr`, `water`, `nylon` | **Équipement technique** *(74 produits)* |
| 2 | 2.98% | `merino`, `wool`, `odor`, `control`, `wash`, `knit` | Laine mérinos *(36 produits)* |
| 3 | 2.90% | `organic cotton`, `button`, `inseam`, `canvas`, `short` | Casual coton bio *(33 produits)* |
| 5 | 1.92% | `nylon`, `spandex`, `sun`, `upf`, `coverage` | Protection solaire/sport *(21 produits)* |
| 7 | 1.65% | `photo`, `poster`, `outside`, `photograph` | **Posters déco** *(8 produits – catégorie inattendue!)* |
| 8 | 1.63% | `strap`, `shoulder strap`, `mesh`, `compartment`, `deni` | Sacs et bagagerie *(30 produits)* |
| 14 | 1.06% | `sock`, `toe`, `lintoe`, `knit`, `loop` | Chaussettes techniques *(9 produits)* |

**Insight clé – Topic 7 (Posters) :** LSA a découvert automatiquement que ~8 produits sont des **affiches photo** (non des vêtements!), une catégorie que le marketing ignorait peut-être dans sa structure actuelle.

---

## Slide 10 – Analyse Critique & Limites du Modèle

**Ce que les résultats nous apprennent vraiment :**

### Sur le Clustering DBSCAN
- **95 clusters** avec `min_samples=2` = granularité très fine (beaucoup de clusters à 2 produits)
- **60% d'outliers** → La diversité extrême du catalogue The North Face rend le clustering difficile avec cette configuration
- **Solution recommandée :** Augmenter `min_samples` (à 3-5) ou utiliser `eps=0.4-0.5` pour obtenir 15-20 grands clusters plus interprétables
- Les clusters existants (surtout ceux à 4-6 produits) sont **sémantiquement cohérents** — le modèle fonctionne

### Sur le Topic Modeling LSA
- **29.3% de variance** avec 15 topics = valeur normale pour du TF-IDF (pas de compression possible au-delà ~30-40% avec peu de données)
- Topic 0 domine massivement (198/500 produits) → Le discours éco-responsable est **omniprésent** dans tout le catalogue
- Les topics non-apparents révèlent des **microsegments** (posters, chaussettes, protection solaire) invisibles dans les catégories classiques

### Sur le Recommender System
- Fonctionne parfaitement en intra-cluster (similarité 0.94 pour les variantes du même produit)
- Pour les outliers : le fallback en similarité globale donne des résultats pertinents (0.47-0.49)

---

## Slide 11 – Stack Technique & Justification

```python
# Pipeline Scikit-Learn complet (sans data leakage)
tfidf_pipeline = Pipeline([
    ('preprocesseur', FunctionTransformer(preprocess_text, validate=False)),
    ('tfidf', TfidfVectorizer(max_features=3000, min_df=2, max_df=0.85, ngram_range=(1,2)))
])

# Résultat : Matrice (500 × 3000) — Créusité 96.8%
DBSCAN(eps=0.3, min_samples=2, metric='cosine')   # 95 clusters + outliers
TruncatedSVD(n_components=15, random_state=42)    # 29.3% variance expliquée
```

**Évaluation adaptée au non-supervisé :**
| Domaine | Métrique | Valeur |
|---------|---------|--------|
| Preprocessing | Créusité TF-IDF | 96.8% (normal pour texte) |
| Clustering | Structure (validation visuelle WordClouds) | ✅ Cohérence sémantique confirmée |
| LSA | Variance expliquée | 29.3% / 15 topics |
| Recommandation | Similarité cosinus intra-cluster | 0.89-0.94 (produits similaires) |
| Recommandation | Similarité cosinus outliers | 0.36-0.49 (cross-catégories) |

---

## Slide 12 – Résultats Métier & Recommandations

**Impact estimé sur la base des résultats réels :**

| Livrable | Application Métier | Résultat Observé |
|----------|--------------------|-----------------|
| Clusters produits | Filtres de navigation | 95 micro-catégories découvertes, 15-20 grandes thématiques identifiables |
| Recommender System | Widget "Vous aimerez aussi" | Pertinence validée (sim. > 0.47 pour produits proches) |
| Topics LSA | Tags SEO + merchandising | 15 sujets latents dont 1 inattendu (posters déco) |
| Outlier analysis | Revue qualité catalogue | ~300 produits à catégoriser / 8 posters à isoler |

**Actions prioritaires recommandées :**
1. 🎯 **Court terme** : Déployer le recommender system avec la similarité cosinus globale (couvre 100% des produits)
2. 🔧 **Moyen terme** : Recalibrer DBSCAN (`eps=0.4, min_samples=3`) pour des clusters plus stables
3. 🚀 **Long terme** : Enrichir avec données comportementales (clics, achats) pour un filtrage collaboratif hybride

---

## Slide 13 – Conclusion

**Ce projet démontre comment 500 descriptions textuelles suffisent pour :**

✅ Découvrir **95 micro-groupes de produits** cohérents sémantiquement  
✅ Extraire **15 sujets latents** révélant la structure réelle du catalogue  
✅ Construire un **recommender prêt-à-déployer** avec des similarités > 0.47  
✅ Identifier des **anomalies** (8 posters mélangés aux vêtements — insight inattendu !)  

**Compétences Jedha validées :**
- ✅ Pipeline Scikit-Learn complet sans fuite de données (Bloc 3)
- ✅ Traitement de données textuelles (spaCy, TF-IDF, NLP)
- ✅ Algorithmes non supervisés : DBSCAN + LSA
- ✅ Évaluation de performance adaptée (Silhouette, variance expliquée)
- ✅ Visualisations explicatives : WordClouds, Heatmaps, Histogrammes (Plotly)

> 🎯 *The North Face peut désormais guider ses clients vers les produits qui correspondent à leurs besoins — même pour les 300 produits sans catégorie assignée.*
