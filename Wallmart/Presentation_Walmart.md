# Détecteur de Ventes pour Walmart - Analyse Prédictive
 
 ![Walmart Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/1024px-Walmart_logo.svg.png)
 
 ---
 
 ## 🛒 Le Problème 
 
 Walmart, le géant américain de la grande distribution, cherche à optimiser la gestion de ses stocks, la planification de son personnel et ses campagnes marketing. Pour y parvenir de manière efficace, la direction a un besoin fondamental : **estimer avec la plus grande précision possible les ventes hebdomadaires de ses différents magasins.**
 
 Jusqu'à présent, ces métriques pouvaient s'appuyer sur de l'instinct ou des statistiques simples, incapables de capter l'influence croisée d'indicateurs économiques divers comme la météo, le coût du carburant ou le taux de chômage.
 
 **Objectif :**
 Construire un modèle d'Intelligence Artificielle (Machine Learning) robuste capable de **prédire automatiquement** le montant des ventes en fonction des signaux extérieurs, tout en évitant le piège du surapprentissage (overfitting).
 
 ---
 
 ## 📊 1. Exploration du Jeu de Données (EDA)
 
 Le jeu de données mis à notre disposition rassemble l'historique de ventes (*Weekly_Sales*) de différents magasins Walmart. 
 
 **Observations clés lors de l'EDA :**
 *   **Valeurs Manquantes :** J'ai du identifier et évincer les lignes pour lesquelles notre cible (le chiffre d'affaires, Y) était absente. Il est techniquement dangereux d'inventer la réponse cible.
 *   **Temporalité :** La date d'origine ("01-01-2010") ne veut rien dire pour une IA classique. J'ai donc décomposé cette variable en composants distincts : Année, Mois, Jour, et JourDeLaSemaine.
 *   **Outliers :** Certaines données externes étaient absurdes ou trop extrêmes. J'ai utilisé l'écart-type ($\mu \pm 3\sigma$) pour écrémer la température, le chômage, l'indice des prix et le coût du diesel.
 
 <br>
 
 ---
 
 ## 🧹 2. Le Preprocessing (Transformation des Données)
 
 Les algorithmes de Machine Learning ne comprennent que des matrices parfaites et numériques.
 
 1. **Les Données Numériques (Température, Prix de l'Essence, Dates, etc.)** :  
    J'ai dû remplacer les valeurs absentes par la *moyenne* globale (`SimpleImputer`), puis j'ai forcé toutes ces métriques à graviter autour d'une échelle commune (`StandardScaler`). Si l'on ne standardise pas, le modèle croit qu'une température de 80 degrés est "plus importante" qu'un taux de chômage de 8%.
 
 2. **Les Données Catégorielles (Identifiant du Magasin, Jour Férié - O/N)** : 
    On remplace les vides par la valeur la plus fréquente, puis on convertit ces "concepts" en colonnes binaires (1 ou 0) avec un `OneHotEncoder`.
 
 3. **Apprentissage Non Supervisé (PCA & K-Means)** : 
    Pour y voir plus clair, j'ai réduit l'immensité du jeu de données en seulement 2 dimensions (Analyse en Composantes Principales). Cette étape a permis de créer un clustering K-Means pour observer visuellement si des magasins ont des "profils" (groupes) de ventes typiques.
 
 <br>
 
 ---
 
 ## 🧠 3. La Modélisation Supervisée : 3 Approches
 
 J'ai ensuite entraîné des modèles pour "apprendre" la règle mathématique qui lie ces entrées au chiffre des ventes :
 
 ### ▶ Baseline : Régression Linéaire Standard
 *   **L'idée** : Trouver la ligne droite parfaite qui passe au centre de toutes nos données.
 *   **Résultat** : Elle fournit un excellent socle sur nos données d'entraînement. Cependant, par sa conception rigide, elle risque le surapprentissage ("Overfitting") et donne potentiellement trop de poids à certains facteurs aléatoires du passé, au lieu de généraliser pour le futur.
 
 ### ▶ Le Modèle Ridge ($L2$)
 *   **L'idée** : Une Régression Linéaire "bridée". Elle pénalise légèrement au carré l'importance globale de toutes les variables (les coefficients) si elles prennent trop la confiance.
 *   **Intérêt** : Pousse l'algorithme à distribuer son importance équitablement plutôt que de devenir dépendant d'une seule colonne phare (ex: La taille du magasin).
 
 ### ▶ Le Modèle Lasso ($L1$)
 *   **L'idée** : Une Régression Linéaire "sévère". Sa pénalité absolue force littéralement les coefficients des données inutiles... **à zéro !**
 *   **Intérêt** : Procède automatiquement à la sélection des caractéristiques (Feature Selection). Si le "taux de chômage" sert à rien à New York, Lasso supprime juste cette variable de l'équation mathématique.
 
 <br>
 
 ---
 
 ## 🏆 4. Optimisation et Résultats
 
 Pour nos modèles Lasso et Ridge, nous devions décider de la **force de la pénalité** (le paramètre `alpha`). 
 Plutôt que de tâtonner au hasard, j'ai utilisé une **GridSearchCV** (Recherche sur Grille par validation croisée). L'ordinateur essaye toutes les forces (ex: 0.1, 1, 10, 100...) et mémorise la performance pour nous livrer le modèle mathématiquement imbattable.

**Résultats obtenus :**
* **Régression Linéaire (Baseline)** : Score R² (Test) = **0.891** (avec un R² d'entraînement de 0.977, un très léger surapprentissage est présent).
* **Ridge (alpha = 0.01)** : Score R² (Test) = **0.892**. La pénalité optimale est faible, apportant une légère amélioration.
* **Lasso (alpha = 500)** : Score R² (Test) = **0.897**. C'est notre meilleur modèle ! En forçant certains paramètres au silence, Lasso généralise mieux l'information pertinente.

### Bilan Walmart
 
 * ✅ Un Pipeline DataFrame complet et sécurisé.
 * ✅ Une IA stabilisée par Régularisation qui n'apprend pas par coeur l'historique mais comprend réellement le système de causalité.
 * ✅ Les coefficients du modèle (Feature Importance) ont été isolés pour prouver aux directeurs locaux ce qui fait réellement fluctuer leur chiffre ce vendredi soir !
