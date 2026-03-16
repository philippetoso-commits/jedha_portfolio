# Uber Pickups - Recommandation de Zones Chaudes

![Uber Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Uber_logo_2018.svg/1024px-Uber_logo_2018.svg.png)

---

## 🚕 Le Problème 

Uber a révolutionné le transport, s'étendant à Uber Eats, Jump Bike, Lime, opérant dans plus de 900 villes pour générer plus de 14 milliards de revenus.

Cependant, il existe un point de friction majeur pour les clients américains (San Francisco, New York...) : **Le temps d'attente.**
Lorsqu'un utilisateur commande une course dans un quartier dynamique (Financial District), il arrive que tous les chauffeurs soient concentrés ailleurs (Castro).

**La contrainte humaine :** Les études d'Uber montrent que si le client attend plus de 5 à 7 minutes, il annule la course (probablement au profit d'un taxi ou d'un concurrent).

**Objectif :**
Créer un puissant algorithme d'Intelligence Artificielle pour **prédire et recommander des Hot-Zones (zones chaudes)** aux chauffeurs, à n'importe quel moment de la journée, pour anticiper la demande avant même qu'elle n'arrive.

---

## 📊 1. Exploration du Jeu de Données (EDA)

Nous avons concentré notre étude sur **New York City** en utilisant un jeu de données public (Uber Trip Data - Avril 2014) contenant l'historique de chaque prise en charge (Latitude, Longitude, Heure).

**Observations clés de l'EDA :**
*   **Temporalité (Feature Engineering) :** La date brute n'est pas interprétable par une machine. J'ai donc décomposé celle-ci pour créer deux nouvelles variables : **Jour de la semaine** et **Heure de la journée**.
*   **Distribution :** Les graphiques montrent une hyper-activité aux heures de pointe (17h - 20h) et de fortes disparités entre un Mardi et un Samedi. Les chauffeurs doivent donc changer de stratégie selon l'heure !
*   **Spatialité :** Un nuage de points dessine naturellement et précisément les contours denses de l'île de Manhattan et des ponts environnants.

<br>

---

## 📍 2. Modélisation Non Supervisée (Clustering)

Puisque la carte nous montre que l'activité est concentrée, nous devions découper l'espace de manière mathématique. Nous ne voulions pas faire de quadrillage arbitraire (carrés), mais épouser le flux du trafic.

1. **L'Algorithme K-Means :**  
   J'ai demandé à l'IA d'observer les coordonnées brutes (Lat/Lon) et d'apprendre par elle-même où se trouvent les **10 épicentres** de la commande New-Yorkaise. Elle a ainsi généré 10 "Hot-Zones".

2. **L'Alternative DBSCAN :**  
   Plus précis mais lourd en calcul, DBSCAN s'appuie sur la **densité**. Au lieu de tracer des zones, il dessine des lignes le long des artères saturées. J'ai utilisé un échantillon stratifié de 10 000 trajets pour l'expérimenter en mémoire RAM.

=> **Résultat :** Chaque course passée est maintenant labellisée avec un numéro de Zone. (Ex: Trajet 542 -> Hot-Zone 3).

<br>

---

## 🧠 3. L'Analyse Prédictive (Apprentissage Supervisé)

Maintenant que nous avons défini les zones géographiquement, l'application Uber doit pouvoir dire au chauffeur ce qu'il doit cibler **maintenant**.

### Le Pipeline de Pré-traitement Scikit-Learn
Avant de nourrir le modèle, le flux de données en temps réel doit passer par un système de tuyauterie ("*Pipeline*") :
*   Transformation des données catégorielles textuelles (ex: "Base d'attache B02512") en langage binaire (1 ou 0) `OneHotEncoder`.
*   Aplatissement / Normalisation de "l'Heure" (0-24) et du "Jour" (0-7) sur un plan de jeu équivalent. Sans le `StandardScaler`, l'Heure écraserait la dimension mathématique du Jour juste parce que son chiffre maximum est 24 !

### Le Moteur : Random Forest Classifier
C'est un algorithme composite (*Ensemble Learning*). Au lieu d'avoir un arbre de décision basique, il en plante une forêt métaphorique complète (50 arbres) qui votent ensemble pour trouver, non-linéairement, quelle heure cible le mieux telle localisation complexe.

<br>

---

## 🏆 4. Résultats et Explicabilité (Feature Importance)

Le score d'exactitude (Accuracy) obtenu confirme mathématiquement que notre modèle capte des motifs horaires fiables pour orienter un chauffeur vers un espace très réduit parmi 10 immenses zones, le tout sans connaître la météo locale !

**L'Explicabilité (Pourquoi le modèle choisit ça ?) :**
Un des atouts majeurs du `RandomForest` est d'ouvrir son algorithme. Nous avons extrait les **Feature Importances** (L'importance des facteurs dans la prise de décision) :
1. **L'Heure (Hour)** : Elle dirige à plus de **63%** le comportement de commande !
2. **Le Jour (DayOfWeek)** : Conduit ~30% de l'influence pour guider le chauffeur vers les endroits commerciaux vs bars/soirées le week-end.
3. **La Base d'où il part** : Influe à moins de 5% sur le lieu final.

### Bilan Stratégique Uber
* ✅ L'invention de zones intelligentes via K-Means / DBSCAN non supervisé.
* ✅ Un flux algorithmique (Pipeline) sécurisé pour l'application en temps réel.
* ✅ Une priorisation temporelle claire pour maximiser la satisfaction client et garder l'attente sous le seuil fatidique des 5 minutes.
