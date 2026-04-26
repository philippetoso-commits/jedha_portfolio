# 📓 Carnet de Bord : Les Coulisses du Projet Kayak

Ce document retrace la réalisation du projet Kayak, de la lecture du brief initial jusqu'au pipeline fonctionnel. L'idée n'est pas de présenter le résultat comme une invention personnelle — le cadre technique était largement balisé par l'énoncé du projet JEDHA — mais plutôt de montrer comment j'ai mis en œuvre les consignes, où j'ai dû m'adapter, et ce que j'ai appris en chemin.

---

## 1. Ce que le brief demandait

Le notebook `01-Plan_your_trip_with_Kayak.ipynb` posait les fondations très clairement. Le projet demandait de :
- Récupérer les coordonnées GPS des 35 villes françaises via **Nominatim** (suggestion directe du brief).
- Collecter les prévisions météo via **OpenWeatherMap** (API et lien fournis dans l'énoncé).
- Scraper les données hôtelières sur **Booking.com** (nom, URL, note, coordonnées GPS, description).
- Stocker le tout dans un **Data Lake S3**, puis nettoyer et charger dans un **Data Warehouse RDS PostgreSQL**.
- Produire deux cartes Plotly : Top 5 destinations et Top 20 hôtels.

En résumé, l'architecture générale, les outils à utiliser et même les sources de données étaient indiqués. Mon travail a consisté à implémenter ce plan, résoudre les problèmes concrets qui sont apparus en cours de route, et faire des choix là où le brief laissait de la liberté.

---

## 2. La météo via API : Suivre le plan (et choisir sa formule)

L'utilisation de Nominatim et d'OpenWeatherMap était directement suggérée par l'énoncé. J'ai donc suivi la démarche proposée :

1. **Géocodage des 35 villes** via Nominatim pour obtenir les coordonnées GPS.
2. **Appels à l'API Forecast** d'OpenWeatherMap pour récupérer les prévisions sur 5 jours par blocs de 3 heures.

Là où j'ai eu un peu de latitude, c'est sur le **calcul du score météo**. Le brief suggérait d'utiliser les champs `daily.pop` et `daily.rain` mais précisait explicitement : *"you can have different opinions on what a nice weather would be like 😎"*. J'ai opté pour une formule simple et lisible :

> **`Score = Température Moyenne − (Pluie Totale en mm × 2)`**

La température comme bonus, la pluie comme pénalité. Rien de révolutionnaire, mais ça a le mérite d'être compréhensible et de produire un classement cohérent (Bormes-les-Mimosas, Cassis, Marseille en tête — sans surprise pour le sud de la France).

---

## 3. Le scraping Booking : Là où la théorie rencontre la réalité

Le brief demandait de scraper Booking.com pour récupérer les données hôtelières, en suggérant même d'extraire les coordonnées GPS directement. Sur le papier, c'est simple. En pratique, c'est là que les choses se sont compliquées.

### Ce que le brief ne mentionnait pas

Booking.com déploie des protections anti-bot très agressives basées sur **AWS WAF**. Concrètement :
- **Challenge JavaScript (Code 202)** : Le serveur détecte les scripts automatisés et renvoie une page de vérification au lieu des résultats. Pour contourner ça, j'ai injecté des cookies de session valides extraits de mon navigateur (`cookie.txt`) dans les requêtes Scrapy.
- **Détection comportementale** : Un scraper trop rapide se fait bannir immédiatement. J'ai dû mettre en place des délais entre les requêtes (`DOWNLOAD_DELAY`) et utiliser des headers HTTP réalistes pour passer sous le radar.

### L'écart avec le plan initial

Le brief suggérait de scraper directement les coordonnées GPS des hôtels. En réalité, Booking masque cette information dans des widgets Google Maps chargés dynamiquement par JavaScript — impossible à capturer avec un scraper statique comme Scrapy. C'est un cas typique où le plan théorique se heurte à la réalité du terrain.

Au final, j'ai pu extraire les données brutes d'environ **880 hôtels** (nom, URL, note, prix, description), mais sans les coordonnées GPS.

---

## 4. Le géocodage a posteriori : S'adapter au terrain

Pour combler l'absence de coordonnées GPS dans les données scrapées, j'ai réutilisé **Nominatim** — déjà suggéré par le brief pour les villes — mais cette fois pour géolocaliser les hôtels eux-mêmes. C'est un pivot que le plan initial ne prévoyait pas, mais qui relève d'une pratique courante en Data Engineering : quand une source ne fournit pas la donnée attendue, on la complète avec une autre.

En soumettant les adresses/noms d'hôtels à l'API, **699 hôtels** sur 880 ont pu être géolocalisés avec précision. Les ~180 restants avaient des adresses trop ambiguës pour être résolues, ce qui est un compromis de qualité acceptable : mieux vaut 699 hôtels bien placés sur la carte que 880 avec des marqueurs erronés.

---

## 5. L'architecture Cloud : Appliquer le schéma demandé

L'architecture **Data Lake (S3) → Data Warehouse (RDS)** était explicitement demandée par l'énoncé. J'ai donc suivi ce schéma :

- **Amazon S3 (Data Lake)** : Stockage des CSV bruts. Au-delà de la consigne, j'ai vraiment apprécié l'utilité de cette couche : vu la fragilité du scraping (cookies qui expirent, structure HTML qui peut changer), avoir des "snapshots" persistants des données brutes est un vrai filet de sécurité.
- **Amazon RDS PostgreSQL (Data Warehouse)** : Stockage des données nettoyées et typées. C'est ici que j'ai appliqué le nettoyage via Regex (par exemple, transformer *"Avec une note de 7.5"* en float `7.5` exploitable) et que les tables `hotels` et `weather` sont prêtes pour des requêtes SQL.

Un détail technique que j'ai découvert en cours de route : pour les cartes Plotly, le style `open-street-map` bloquait l'affichage des tuiles à cause de restrictions de Referrer Policy dans Jupyter. J'ai basculé vers `carto-positron`, qui fonctionne sans accroc et offre un rendu plus épuré.

---

## 6. Ce que j'en retiens

Ce projet m'a permis de mettre en pratique un pipeline de Data Engineering de bout en bout. Le brief fournissait un cadre solide — les outils, les sources, l'architecture — mais l'implémentation réelle m'a confronté à des problèmes concrets que l'énoncé ne couvrait pas :

- **Le contournement des protections anti-bot** de Booking (cookies, headers, délais).
- **L'adaptation du plan** quand les coordonnées GPS n'étaient pas accessibles via le scraping.
- **Les choix techniques quotidiens** : quelle formule de scoring ? Comment gérer les données manquantes ? Quel style de carte choisir pour éviter les bugs d'affichage ?

Si je devais nommer la leçon principale : un plan, aussi détaillé soit-il, n'est qu'un point de départ. La valeur ajoutée d'un Data Engineer se mesure dans sa capacité à s'adapter quand la réalité ne colle pas à la théorie.
