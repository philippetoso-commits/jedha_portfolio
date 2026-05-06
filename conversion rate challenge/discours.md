# Discours : Conversion Rate Challenge - Stratégie ADN (5 minutes)
*Ce script est conçu pour une présentation percutante de 5 minutes. Il met l'accent sur votre victoire au Hackathon et votre approche innovante.*

---

## [0:00 - 1:00] L'Accroche & Le Challenge
**Action :** Afficher la slide de titre avec le logo "1ère Place Leaderboard".

"Bonjour à tous. Aujourd'hui, je vais vous parler de **performance pure**. 

Dans le cadre de ma certification chez Jedha, j'ai participé à un challenge de type Kaggle. L'objectif : prédire le taux de conversion des utilisateurs d'un site e-commerce. Sur plus de 30 participants, j'ai eu l'honneur de terminer à la **1ère place du classement final**.

Mais ce qui est intéressant, ce n'est pas seulement le résultat, c'est la méthode. Là où beaucoup ont cherché la complexité, j'ai cherché l'**ADN des données**."

---

## [1:00 - 2:00] L'Exploration (EDA) : L'Évidence
**Action :** Afficher un graphique montrant la corrélation entre `Total_Pages_Visited` et la conversion.

"En analysant les données, une vérité a sauté aux yeux. La variable `Total_Pages_Visited` n'est pas juste un indicateur, c'est un prédicteur quasi-parfait : au-delà de 15 pages vues, le taux de conversion frôle les 100%.

À l'inverse, j'ai détecté des 'bruits' statistiques : des utilisateurs très jeunes ou très âgés qui convertissaient de manière erratique. Mon défi était de créer un modèle qui capture la tendance lourde sans se laisser distraire par ces anomalies."

---

## [2:00 - 3:30] La Stratégie "ADN" (Le Secret de la Victoire)
**Action :** Afficher les paramètres du XGBoost et les contraintes de monotonie.

"Pour gagner, j'ai testé des modèles très complexes : du Stacking, du Voting, des réseaux de neurones. Mais j'ai fini par tout jeter pour revenir à une approche que j'appelle **'ADN'**.

J'ai utilisé un unique modèle **XGBoost**, mais je l'ai bridé mathématiquement avec des **Contraintes de Monotonie**. 
J'ai forcé le modèle à obéir à des lois physiques : 
1. Plus tu visites de pages, plus la probabilité *doit* monter.
2. Plus tu es âgé, plus elle *doit* descendre.

En empêchant le modèle de 's'inventer des histoires' sur les cas particuliers, il est devenu incroyablement robuste sur les données qu'il n'avait jamais vues (le test set)."

---

## [3:30 - 4:30] L'Optimisation de Seuil
**Action :** Afficher la courbe de F1-Score en fonction du seuil.

"Enfin, la dernière étape cruciale a été l'**optimisation du seuil**. Par défaut, un modèle dit 'Oui' à 0.5 de probabilité. 
Mais notre métrique de succès était le **F1-Score**. En faisant varier ce seuil, j'ai trouvé que le point d'équilibre optimal était à **0.385**. 

Ce réglage fin m'a permis de maximiser le rappel sans sacrifier la précision, me propulsant en tête du leaderboard avec un score de **0.7658**."

---

## [4:30 - 5:00] Conclusion Business
**Action :** Afficher les 3 recommandations clés.

"Pour conclure, ce projet prouve que la Data Science n'est pas une course à l'armement technologique. En comprenant la 'physique' de son business et en appliquant des contraintes métier à ses algorithmes, on obtient des modèles plus simples, plus explicables et, au final, plus performants.

Je vous remercie, et je suis prêt pour vos questions."
