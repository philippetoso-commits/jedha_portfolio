# FAQ - Présentation Walmart Sales Detector
 
 Cette FAQ regroupe les questions probables de l'audience ou du jury concernant les choix d'architecture Machine Learning du projet.
 
 ---
 
 ### Q1 : Pourquoi avoir décidé de séparer en Pandas l'exploration et les anomalies, et en Scikit-Learn l'encodage ?
 
 **Réponse attendue :** 
 Pandas est notre outil de chirurgien (le *Data Wrangling*). Lorsque j'ai dû supprimer des lignes absurdes (valeurs extrêmes / Outliers) ou des lignes où je ne connaissais pas le CA demandé (`Weekly_Sales`), je change la morphologie, le nombre de lignes au cœur de notre base d'apprentissage.
 Scikit-Learn (`Pipeline`, `ColumnTransformer`) n'est là que pour la traduction algorithmique "à la volée". Si je devais supprimer les lignes avec le Pipeline Sklearn, à l'avenir quand un magasin Walmart nous demandera de faire une prédiction sur le jour en cours et qu'une valeur manque, le modèle Sklearn crasherait ou refuserait de prédire au lieu de transformer silencieusement !
 
 ### Q2 : N'est-ce pas dangereux d'enlever des données via la règle des "$3\Sigma$" (moyenne +/- 3 Ecarts-Types) ? On perd de la donnée précieuse non ?
 
 **Réponse attendue :**
 En statistique gaussienne, un écart-type de plus de 3 englobe 99.7% du comportement d'un jeu de données naturel. Supprimer les éléments restants qui échappent à cela (0.3% d'Outliers) ne modifie pas l'ADN de l'apprentissage de Walmart. À l'inverse, si nous mettions à notre Régression Linéaire un mois où l'essence valait 50$ (soit suite à une frappe au clavier ratée, soit une anomalie géopolitique), la ligne "attirerait" toute la régression autour d'elle, biaisant gravement les prévisions du futur quotidien.
 
 ### Q3 : À quoi servent "réellement" l'algorithme PCA et K-Means vis-à-vis des Ventes ?
 
 **Réponse attendue :**
 L'algorithme des Ventes est très abstrait, il y a 8 à 10 colonnes, c'est un tableau infini impossible à visionner pour le cerveau humain qui n'est fait qu'en 3D. 
 La `PCA` (Apprentissage Non Supervisé de réduction de dimension) agit comme la "compression géographique" de nos colonnes pour les dessiner en 2 Variables Fantômes (X et Y) sur un graphe 2D.
 L'algorithme de Clustering `K-Means`, projeté sur cette PCA, permet d'afficher en couleur les différents segments de magasins naturels. Cela permet de répondre à la question : "Les succursales Walmart de Californie et de New York se ressemblent-elles de l'oeil de la machine indépendamment de leurs chiffres ?".
 
 ### Q4 : Comment la StandardScaler protège-t-elle le coefficient des variables ?
 
 **Réponse attendue :**
 Si je ne "standardise" pas, la température évoluera entre mettons -10° et 30°, tandis que le taux de chômage variera entre 0.05 et 0.15. Un modèle mathématique pur considérera faussement que la variable `Température` possède 300x plus d'inertie (de variance) que le chômage de par sa grandeur intrinsèque. La régression ne considèrera que la température et tuera le chômage ! Le `StandardScaler` recentre de force toutes les composantes du monde autour d'une moyenne de zéro et un écart-type de un. Les modèles jouent à chances égales.
 
 ### Q5 : Pourquoi s'infliger RIDGE ou LASSO au lieu de régler une Baseline Linear Regression correctement en manipulant soi-même les colonnes ?
 
 **Réponse attendue :**
 Car la manipulation manuelle des colonnes est biaisée par l'intuition humaine et inexploitable à grande échelle. La `Régression Linéaire` standard calcule ses poids sans aucune contrainte d'apprentissage (Ordinary Least Squares). Elle "Overfit" (sur-apprend) ses courbes sur les moindres hoquets de l'historique historique du jeu Train.
 Imposer une pénalisation automatisée (norme *L1* de Lasso qui ampute les mauvais élèves à 0, et *L2* de Ridge qui lisse l'audace globale) est mathématiquement le seul garde-fou connu des systèmes statistiques linéaires pour ne pas confondre Bruit Temporel aléatoire et Signal Causal de Vente.
 
 ### Q6 : C'est quoi concrètement GridSearchCV et la validation croisée (Cross Validation) utilisés sur la fin ?
 
 **Réponse attendue :**
 Pour les algorithmes complexes, on ajoute le paramètre "Alpha" (la force de l'amende infligée par la Régularisation Ridge ou Lasso). Si on décide soi-même que l'Alpha = 4, on triche parce qu'on modifie le code en fixant notre test.
 `GridSearchCV` instancie automatiquement l'apprentissage en créant 5 petites sous-divisions de classe de notre historique Walmart (Cross Validation avec `cv=5`). Il va tester Alpha=1 sur les 4 partitions en révisant sur la cinquième. Puis inverser. Il fera cela pour 0.1, 1, 10, etc., en calculant sans cesse le R² Score à chaque fois, pour nous fournir statistiquement la formule absolue sans aucune manipulation de notre part.
