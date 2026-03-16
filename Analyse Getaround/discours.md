# GetAround — Script de Discours (10 minutes)
## Jedha Bootcamp — Bloc 5

---

### [00:00 – 00:45] Introduction & Contexte

Bonjour à tous !

Aujourd'hui je vais vous présenter le projet **GetAround**, que j'ai réalisé dans le cadre du Bloc 5 du bootcamp Jedha, dédié à l'**industrialisation des algorithmes de Machine Learning**.

GetAround, c'est l'Airbnb des voitures. Une plateforme de location entre particuliers, active depuis 2009, qui compte aujourd'hui plus de 5 millions d'utilisateurs et environ 20 000 voitures disponibles dans le monde.

Ce projet a **deux volets distincts**, mais complémentaires :

- Le premier est une **analyse de données** : aider le Product Manager à choisir un délai minimum entre deux locations pour réduire les frictions liées aux retards.
- Le second est un **module Machine Learning** : construire un modèle qui suggère le prix journalier optimal d'un véhicule aux propriétaires.

---

### [00:45 – 02:00] Les Données

Pour ce projet, j'ai travaillé avec deux datasets :

Le premier — `get_around_delay_analysis.xlsx` — contient **21 310 locations** avec pour chaque location des informations sur le type de checkin — mobile ou Connect — l'état de la location — terminée ou annulée — et surtout le **retard à la restitution en minutes**.

Ce qui est clé, c'est que **1 841 locations seulement** ont une location précédente sur le même véhicule. Ce sont les seuls cas où un retard peut réellement impacter le conducteur suivant.

Le deuxième — `get_around_pricing_project.csv` — contient **4 843 véhicules** avec 13 features : 2 numériques comme le kilométrage et la puissance moteur, 4 catégorielles comme la marque et le type de voiture, et 7 booléennes comme le GPS ou la climatisation. La variable cible est le **prix journalier de location**, avec une médiane à 119 euros.

---

### [02:00 – 03:30] Analyse Exploratoire — Retards

Passons à l'EDA.

Sur les locations **terminées**, j'ai constaté que **plus de la moitié** présentent un retard de restitution. C'est significatif. Mais attention : la plupart de ces retards n'ont **aucun impact** car il n'y a pas de prochaine location prévue sur le même véhicule.

Le vrai enjeu, ce sont les **1 841 cas consécutifs**. J'ai identifié parmi eux les cas **problématiques** : ceux où le retard dépasse le délai disponible entre les deux locations. Ces cas correspondent aux frictions réelles vécues par les clients — attente, annulation.

J'ai visualisé ces distributions avec des histogrammes et des box plots Plotly par type de checkin. Résultat : les voitures Connect ont une dispersion de retard légèrement plus élevée que les voitures Mobile.

---

### [03:30 – 05:00] Simulation du Seuil — Recommandation Business

Pour aider le Product Manager, j'ai simulé **l'impact de différents seuils** — de zéro à 720 minutes — sur deux métriques clés :
- Le pourcentage de **problèmes résolus**
- Le pourcentage de **locations affectées** — autrement dit, le revenu potentiellement perdu

Le résultat s'affiche sous forme d'une courbe de **trade-off** en double axe Y. Et ma recommandation est claire :

**Un seuil de 60 minutes, appliqué à toutes les voitures**, résout environ **47 % des cas problématiques** pour un impact revenu inférieur à **2 % des locations**. C'est le meilleur équilibre.

Si on veut monter à 120 minutes, on résout 67 % des problèmes, mais on augmente l'impact revenu à plus de 3 %.

Alternative possible : limiter aux **voitures Connect uniquement**, ce qui réduit de moitié l'impact sur le revenu tout en protégeant le segment le plus sensible.

---

### [05:00 – 06:30] EDA Pricing & Feature Engineering

Pour le modèle de pricing, j'ai d'abord analysé les corrélations.

Sans surprise, la **puissance moteur** est le facteur le plus corrélé au prix — coefficient de corrélation autour de 0,55. Le kilométrage joue légèrement à la baisse. Et les équipements comme le GPS ou la connexion GetAround ajoutent un premium d'environ 10 euros par jour.

Par type de voiture : les **cabriolets et coupés** sont les plus chers, les citadines les moins chères. Par marque : Porsche et Maserati se démarquent fortement.

Pour le preprocessing, j'ai mis en place un **pipeline Scikit-Learn** avec un `ColumnTransformer` :
- `StandardScaler` sur les variables numériques
- `OneHotEncoder` sur les catégorielles
- Passthrough direct sur les booléennes

Ce pipeline garantit l'absence de **data leakage** entre train et test.

---

### [06:30 – 08:00] Modélisation Machine Learning & MLflow

J'ai testé **3 modèles** en régression supervisée :

D'abord la **régression linéaire** comme baseline — Test R² autour de 0,54, ce qui est faible mais constitue un point de référence.

Ensuite un **Random Forest** optimisé par `GridSearchCV` sur 5 folds croisés. Résultat : Test R² de **0,73**, RMSE d'environ **16,9 €/jour**.

Enfin un **Gradient Boosting** — performances très proches, autour de 0,79 mais sur une baseline de test différente dans mes expérimentations.

J'ai tracké toutes ces expériences dans **MLflow** — paramètres, métriques, et modèles sérialisés. C'est ce qui valide l'aspect industrialisation du Bloc 5.

Le modèle final — RandomForest — a été sérialisé avec `joblib` et déployé dans l'API.

---

### [08:00 – 09:00] Architecture de Production

L'API est construite avec **FastAPI**, qui génère automatiquement la documentation Swagger à `/docs` — une exigence du projet.

Elle expose **4 endpoints** :
- `GET /` — message de bienvenue
- `GET /health` — monitoring opérationnel
- `GET /cars/stats` — statistiques du dataset pour contextualiser une prédiction
- `POST /predict` — prédiction du prix pour une seule voiture
- `POST /predict/batch` — pour plusieurs voitures d'un coup

Toutes les entrées et sorties sont en **JSON nommé** — lisible et documenté.

L'ensemble est **containerisé avec Docker** et déployé sur **Hugging Face Spaces** via le SDK Docker. L'URL est `huggingface.co/spaces/philippetos/GetAround`.

Le **dashboard Streamlit** est également déployé sur Hugging Face — côté Spaces SDK Streamlit.

---

### [09:00 – 10:00] Conclusion & Questions

Pour résumer :

- J'ai livré **deux notebooks** complets — anglais et français — avec 17 graphiques Plotly et une pipeline ML complète trackée par MLflow.
- Un **dashboard interactif** pour le Product Manager — seuil et périmètre configurables en temps réel.
- Une **API en production** with documentation auto-générée.
- Une **infrastructure Dockerisée** déployée sur Hugging Face.

Le modèle final atteint un **R² de 0,73** avec une erreur moyenne de **11 euros** — ce qui en fait un outil concret et utilisable pour guider les propriétaires dans la fixation de leur prix.

Je vous remercie pour votre attention — je suis prêt pour vos questions !
