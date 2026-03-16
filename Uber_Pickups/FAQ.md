# FAQ - Présentation Uber Pickups (Hot Zones)

Cette FAQ regroupe les questions probables de l'audience ou du jury concernant les choix d'architecture Machine Learning du projet.

---

### Q1 : Pourquoi utiliser le Machine Learning (K-Means) pour faire des zones, plutôt que de simplement découper NYC en un quadrillage régulier ?

**Réponse attendue :** 
Un quadrillage manuel (par exemple faire des carrés de 1km²) est arbitraire et ignore totalement la dynamique de la ville. Certains carrés en plein océan ou au milieu d'un parc n'auront aucune commande, tandis que d'autres à Times Square seront surchargés. Le clustering par **K-Means** positionne mathématiquement ses centres (centroïdes) de façon à minimiser la distance avec les commandes réelles. Les clusters "épousent" donc la densité organique de la demande.

### Q2 : Pourquoi avez-vous utilisé DBSCAN en plus de KMeans, et pourquoi l'avoir lancé sur un sous-échantillon ?

**Réponse attendue :**
Contrairement à K-Means qui forme des sphères presque parfaites, DBSCAN crée des clusters de n'importe quelle forme géométrique en se basant sur des points très denses reliés entre eux. C'est idéal pour tracker des avenues congestionnées de long en large.
Cependant, la complexité mathématique de DBSCAN est très lourde (souvent $O(n^2)$). Calculer les distances entre 100 000 points simultanément saturerait la mémoire (RAM) d'un ordinateur classique. J'ai donc dû l'exécuter sur un sous-échantillon représentatif de 10 000 lignes pour préserver la performance du notebook.

### Q3 : C'est très rare de voir de l'Apprentissage Non-Supervisé ET Supervisé en même temps. Pourquoi cette architecture ?

**Réponse attendue :**
C'est effectivement un pipeline composite ! Le jeu de données initial d'Uber n'avait pas de variable "Cible" (Y). On ne nous disait pas "Cette course est une course premium" à prédire. 
Nous avons d'abord utilisé l'algorithme Non Supervisé (le Clustering) pour **créer artificiellement notre variable Cible**, c'est-à-dire l'attribut `HotZone`. Une fois que chaque ligne de notre historique possédait son label de Hot-Zone, nous avons pu basculer en Apprentissage Supervisé, où un Random Forest a appris à prédire cette zone finale à un instant T (X).

### Q4 : Pourquoi avoir choisi un algorithme complexe de `RandomForestClassifier` plutôt qu'une simple `Regression Logistique` pour la prédiction ?

**Réponse attendue :**
La relation entre nos variables et la zone de distribution géographique n'est pas linéaire. Par exemple, ce n'est pas parce que l'Heure avance (1h, 2h, ... 18h) que la latitude et la longitude de la course avancent aussi de façon droite ! 
La **Régression Logistique** aurait très mal géré ces relations non-linéaires et circulaires (les heures qui bouclent, les dynamiques cycliques). Le modèle **Random Forest (Forêt Aléatoire)** découpe les règles de décision en milliers de branches (ex: *Si Heure > 17h ET Jour = Vendredi, alors...*) ce qui est infiniment plus adapté pour relier du temps à une localisation !

### Q5 : Pourquoi avez-vous encapsulé le StandardScaler et le OneHotEncoder dans un `Pipeline` et `ColumnTransformer` ? Ne pouvait-on pas utiliser `.apply()` ou `.replace()` ?

**Réponse attendue :**
Si l'on fait les transformations manuellement dans Pandas avant le Split (comme appliquer une moyenne pour remplir les vides), la donnée Test influence secrètement la donnée Train (Fuite de Données ou *Data Leakage*). 
En utilisant un objet `Pipeline` Scikit-Learn, ce dernier "apprend" (`.fit()`) les échelles et catégories strictement et uniquement sur les données d'entraînement (`X_train`), et l'applique aveuglément sur le Test. De plus, pour passer l'algorithme en production dans l'application Uber, ce tuyau unique prendra les coordonnées en direct du smartphone sans crasher.

### Q6 : L'accuracy (précision globale) de la prédiction du Random Forest semble parfois tourner autour de 30-40%. Est-ce un échec du modèle ?

**Réponse attendue :**
Non, absolument pas un échec, c'est contextuel. Nous tentons de prédire 10 classes (Zones) possibles au coude-à-coude sur des paramètres minuscules (Il n'y a que l'heure et le jour en entrée !). Un aléatoire parfait nous donnerait 10% de chance. Doubler ou tripler ce score juste par la connaissance du temps est une énorme victoire contextuelle. En ajoutant la météo et l'événementiel, cette *accuracy* exploserait. Même avec ce score, l'envoi d'un chauffeur réduit tout de même la probabilité qu'il soit dans une zone totalement morte.
