# Discours : Walmart Sales Prediction (10 minutes)
 
 *Ce document est un script guidé pour accompagner votre présentation. Vous pouvez vous l'approprier avec votre propre style.*
 
 ---
 
 ## 🕒 [0:00 - 1:30] Introduction & Contexte
 
 **Action :** Afficher la première slide (Le Problème Walmart)
 
 "Bonjour à tous. Le projet que je vous présente aujourd'hui concerne l'analyse prédictive appliquée au géant du retail : **Walmart.** 
 Si vous êtes gérant d'un supermarché, votre hantise c'est le gâchis des denrées périssables, ou à l'inverse, la rupture de stock un samedi d'affluence. Pour éviter cela, on aurait le choix entre tirer les cartes du tarot, ou s'appuyer sur la Science des Données.
 
 J'ai donc traité ce problème en créant un Modèle d'Intelligence Artificielle de Machine Learning. Il doit lire les conditions économiques de la semaine (Est-ce que c'est les vacances ? Quel temps fait-il ? A quel prix est le diesel pour les camions ?) et en déduire le chiffre d'affaires (les Ventes) au dollar près."
 
 ---
 
 ## 🕒 [1:30 - 3:00] Nettoyage et EDA
 
 **Action :** Défiler sur le notebook la partie EDA et le nettoyage pandas.
 
 "Avant de lancer la moindre modélisation, je dois scruter nos données historiques : l'Exploratory Data Analysis. 
 
 Tout d'abord, j'ai purgé les lignes absurdes. Par exemple, si nous ignorons le chiffre des ventes (notre cible), il est impératif de supprimer cette ligne entière au lieu de l'imputer, sous peine de biaiser l'IA.
 J'ai scruté notre jeu et retiré ce que l'on appelle les *Outliers* mathématiques, c'est-à-dire les aberrations météos ou de prix qui surviennent au-delà des 3 écarts-types de la moyenne des données. 
 
 Autre problème majeur : l'ordinateur de Walmart ne comprend pas le texte des dates (comme '05-02-2010'). J'ai dû briser et séparer cette colonne en 'Année', 'Mois', 'Jour'. C'est ce qu'on appelle du *Feature Engineering*."
 
 ---
 
 ## 🕒 [3:00 - 4:30] L'Apprentissage Non Supervisé
 
 **Action :** Montrer les graphiques de la PCA et des K-Means.
 
 "Durant mon analyse, j'ai utilisé des modèles dits 'Non Supervisés'. Ils naviguent en aveugle pour nous trouver des règles cachées dans les magasins Walmart.
 D'abord, via une PCA (Analyse en Composantes Principales), j'ai écrasé l'immense nombre de variables en seulement deux dimensions pour obtenir une carte.
 
 Ensuite, j'y ai appliqué un algorithme de *K-Means* pour colorer mes nuages de points. Ainsi, je peux segmenter de manière très visible que les magasins se scindent en groupes homogènes (ex: les petits commerces, les géants). Cela aide le directeur marketing à savoir qu'il vend bien à *X* ou *Y* profils de boutiques distinctes."
 
 ---
 
 ## 🕒 [4:30 - 6:00] La Pipeline et La Régression Linéaire (Baseline)
 
 **Action :** Sur le notebook, pointer le ColumnTransformer et lancer la LinearRegression
 
 "Il est temps de passer à l'Apprentissage Supervisé. J'ai utilisé l'arsenal `Scikit-Learn` pour construire un tuyau de prétraitement (`Pipeline`). Il impute en temps réel les données manquantes, traduit les jours fériés en un langage machine (OneHotEncoder) et standardise obligatoirement nos mètres et nos degrés météo sur la même échelle de variance (`StandardScaler`).
 
 Une fois prêtes, ces données sont injectées dans mon tout premier modèle expérimental : la Régression Linéaire Baseline.
 Ce basique de l'IA tire un trait mathématiquement pur dans notre nuage pour prévoir les ventes. Sa performance (le R² score) est excellente sur la théorie... mais en réalité, ce modèle est un élève bête qui 'apprend par coeur' l'historique du cours. Face aux nouvelles pandémies ou nouveaux hivers, il risque cruellement de se tromper. C'est le fameux Phénomène d'**Overfitting**."
 
 ---
 
 ## 🕒 [6:00 - 8:00] Éviter l'Overfitting : La Régularisation 
 
 **Action :** Descendre vers Ridge et Lasso avec GridSearchCV.
 
 "Pour forcer l'IA à être robuste dans ses prédictions et combattre cet overfitting, j'invoque la Régularisation mathématique.
 - Le **Modèle RIDGE** ajoute un malus à l'intelligence si elle accorde des poids démesurés à de petites variables. Il équilibre l'apprentissage démocratiquement.
 - Le **Modèle LASSO** est drastique. Il agit en ciseaux automatiques. Si le taux de chômage en été n'influence finalement pas tant les ventes d'un magasin, Lasso compresse son coefficient de calcul littéralement à 0% !
 
 Mais du coup, à quel point devais-je 'imposer' ces malus ($Alpha$) ? Impossible de le deviner soi-même. J'ai donc appelé un superviseur externe (`GridSearchCV`). Il a instancié des centaines de versions de notre test via *Cross-Validation* (validation croisée) pour statuer formellement sur le méta-paramètre gagnant universel.

**Action :** Montrer les scores finaux obtenus.

"Et les résultats parlent d'eux-mêmes ! Notre modèle de base atteignait une précision (R²) de 89.1% sur des données inconnues. Ridge, avec un malus ciblé à `alpha=0.01`, monte timidement à 89.2%. Mais le grand vainqueur est Lasso : avec une pénalité forte de `alpha=500`, il parvient à réduire au silence le bruit inutile pour atteindre 89.7% de réussite en Test. C'est ce dernier modèle que nous retiendrons."

---
 
 ## 🕒 [8:00 - 10:00] Conclusion Stratégique
 
 **Action :** Revenir sur les coefficients des Features Importance (la fin de la Linear Reg)
 
 "Finalement, il est crucial de comprendre que faire de l'IA pour l'IA chez Walmart, ça ne sert à rien.
 L'atout final de cette modélisation, que les réseaux de neurones complexes n'ont pas forcément, c'est l'**Explicabilité**.
 
 Chez Walmart, la direction peut ouvrir les coefficients de ce modèle Ridge et lire texto les poids. La direction saura si c'est la Température, le prix du fioul, ou l'indicateur de Jour de l'An qui mène les ventes à la relance ! 
 Nous disposons donc à présent d'un pipeline complet qui peut être injecté dans tous les hypermarchés mondiaux pour dimensionner la distribution de demain.
 
 Merci de votre écoute, l'équipe Data et moi-même sommes prêts à répondre à vos questions."
