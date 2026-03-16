# Discours : Uber Pickups - Recommandation de Hot-Zones (10 minutes)

*Ce document est un script guidé pour accompagner votre présentation. Vous pouvez vous l'approprier avec votre propre style.*

---

## 🕒 [0:00 - 1:30] Introduction & Contexte

**Action :** Afficher la première slide (Problématique Uber)

"Bonjour à tous. Le projet que je vous présente aujourd'hui a été réalisé pour l'une des startups les plus influentes au monde : **Uber.**
Uber a révolutionné le transport urbain, mais ils font face à un point de friction majeur : les temps d'attente. Si un utilisateur commande une course dans le Financial District à San Francisco et que tous les chauffeurs tournent à vide vers Castro, l'utilisateur risque de s'impatienter. S'il attend plus de 5 à 7 minutes, la probabilité qu'il annule sa course explose.

L'équipe Data d'Uber nous a donc missionné pour **aider les chauffeurs à se positionner strategiquement** dans la ville, avant même que les clients ne commandent !
J'ai conçu une série d'algorithmes de Machine Learning pour la ville de New York, capables de recommander les *Hot-Zones* idéales en fonction de l'heure et du jour de la semaine."

---

## 🕒 [1:30 - 3:30] Nettoyage et Exploration (EDA)

**Action :** Défiler sur le notebook la partie EDA et l'extraction temporelle.

"Afin de guider notre IA, il fallait d'abord comprendre les habitudes des New-Yorkais. J'ai utilisé un historique massif de courses d'Avril 2014, contenant les coordonnées géographiques exactes (Latitude et Longitude) de chaque prise en charge. J'ai échantillonné ce volume à 100 000 lignes pour garantir un calcul fluide en temps réel.

En explorant ces datas (EDA), j'ai fait face à un premier obstacle : la machine ne lit pas un format date naturel. J'ai appliqué du *Feature Engineering* pour disséquer la colonne temporelle et extraire deux informations cruciales : **l'Heure de la journée** et le **Jour de la Semaine**.
En visualisant ces données, des tendances claires sont apparues : les commandes explosent en fin d'après-midi, et subissent des fluctuations massives en fonction du jour (ex: Vendredi soir vs Lundi matin). Enfin, sur le nuage de points spatial, la densité d'activité autour de Manhattan sautait littéralement aux yeux."

---

## 🕒 [3:30 - 6:00] Apprentissage Non Supervisé : Dessiner les Hot-Zones

**Action :** Montrer la carte Plotly interactive et les résultats du clustering.

"Mais dire à un chauffeur 'Allez sur Manhattan' est bien trop vague. Comment l'ordinateur peut-il tracer mathématiquement des zones intéressantes sans que nous les dessinions manuellement ?
J'ai fait appel à l'**Apprentissage Non Supervisé (Clustering)**. Ces modèles naviguent à l'aveugle pour regrouper les données similaires :

- Dans un premier temps, j'ai utilisé **K-Means**, en lui demandant de forger 10 'Centres' d'attraction géographiques répartis de manière équilibrée sur le trafic réel.
- J'ai également expérimenté **DBSCAN**, un modèle redoutable basé sur la *densité*. Au lieu de créer des cercles parfaits, il trace des zones épousant réellement les embouteillages denses.

Ce processus a permis de labelliser mathématiquement notre carte en une dizaine de **Hot-Zones distinctes**."

---

## 🕒 [6:00 - 8:30] L'Apprentissage Supervisé : L'Analyse Prédictive

**Action :** Sur le notebook, pointer le ColumnTransformer et lancer le Random Forest

"Nous avons donc nos zones sur la carte. Très bien. Mais la mission est de dire au chauffeur *où* aller *maintenant* !
Nous entrons dans l'**Apprentissage Supervisé**. J'ai développé un Modèle capable de prédire cette fameuse Hot-Zone cible, uniquement en lisant l'heure et le jour actuel.

Pour que ce modèle soit déployable en conditions réelles, j'ai bâti un **Pipeline Scikit-Learn**. Il s'occupe de :
1. Normaliser l'Heure et le Jour pour qu'aucun ne domine mathématiquement l'autre (StandardScaler).
2. Encoder la Base d'expédition Uber (variable textuelle) en format binaire (OneHotEncoder).

Ces données transformées nourrissent ensuite un algorithme avancé : le **Random Forest Classifier (Forêt Aléatoire)**. Plutôt qu'une simple régression qui serait perdue face à la subtilité du trafic new-yorkais, cet algorithme agrège 50 arbres de décisions différents pour statuer sur la meilleure zone géopolitique à viser !"

---

## 🕒 [8:30 - 10:00] Évaluation et Conclusion Stratégique

**Action :** Revenir sur les Features Importances.

"Quelle est l'efficacité de notre recommandation ? Notre modèle affiche ses meilleures prédictions (Accuracy) en s'appuyant très majoritairement sur les bonnes informations. 
Le point le plus fort d'un Random Forest, c'est son **Explicabilité**. En extrayant l'importance de nos variables, on s'aperçoit que l'Heure ('Hour') pilote plus de 63% de la décision de zone, contre environ 30% pour le Jour de la semaine. La base d'où part le chauffeur n'a curieusement que peu d'impact.

Aujourd'hui, nous livrons à Uber un outil en deux phases : un radar géospatial pour dessiner les zones dynamiquement (Non Supervisé), couplé à un moteur prédictif temporel (Supervisé) pour y envoyer les chauffeurs au meilleur moment.
La réduction estimée du temps d'attente pour le client final promet d'être drastique.

Je vous remercie, et je suis prêt à répondre à vos questions."
