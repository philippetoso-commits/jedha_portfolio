# Questions Fréquentes — Projet Kayak ✈️

Ce document répond aux questions qu'on pourrait se poser en parcourant le projet. J'ai essayé d'être transparent sur mes choix, y compris ceux qui ne sont pas parfaits.

---

### Pourquoi ne pas avoir scrapé directement les coordonnées GPS depuis Booking.com ?

Le brief (`01-Plan_your_trip_with_Kayak.ipynb`) suggérait effectivement de récupérer la latitude et la longitude de chaque hôtel lors du scraping. En pratique, Booking ne les affiche pas dans le HTML statique : les coordonnées sont chargées dynamiquement par JavaScript dans un widget Google Maps, ce qui les rend invisibles à un scraper comme Scrapy.

J'ai donc fait en deux temps :
1. **Scraping** (`booking avec description.ipynb`) : récupération de tout ce qui est accessible dans le DOM statique (nom, URL, note, prix, description).
2. **Géocodage** (`geocodageafterscrap.ipynb`) : passage des adresses dans l'API Nominatim pour obtenir les coordonnées GPS a posteriori.

C'est une adaptation que le brief ne prévoyait pas, mais c'est une approche classique en Data Engineering quand une source ne fournit pas la donnée attendue.

---

### Pourquoi Booking.com bloque-t-il le scraping ? Comment ça a été contourné ?

Le brief ne le mentionnait pas, mais Booking utilise un système anti-bot très agressif basé sur **AWS WAF**. Dès qu'un robot est détecté, le serveur renvoie une page de challenge JavaScript (code 202) au lieu des résultats.

Voici les techniques utilisées pour passer :
- **Cookies de session valides** extraits d'un navigateur réel (`cookie.txt`), injectés dans chaque requête Scrapy.
- **User-Agent réaliste** et headers HTTP imitant Chrome.
- **Délai entre les requêtes** (`DOWNLOAD_DELAY`) pour éviter le bannissement.

Important à noter : ces techniques sont fragiles. Les cookies expirent au bout de quelques heures, et c'est pour ça que le CSV de résultats (`booking_full_data.csv`) a été sauvegardé comme "snapshot" de référence sur S3.

---

### Pourquoi y a-t-il deux fichiers CSV pour les hôtels ?

Le fichier `booking_full_data-save.csv` est une sauvegarde manuelle faite avant une phase de nettoyage. C'est une habitude un peu artisanale (dans un projet industriel, on utiliserait du versioning de données), mais ça a fait office de filet de sécurité.

Ce fichier peut être supprimé sans impact — c'est `booking_final_gps.csv` (enrichi avec les coordonnées GPS) qui est utilisé en aval.

---

### Pourquoi l'ETL est-elle répartie sur deux notebooks ?

Bonne question, et la réponse honnête c'est que ce n'est pas idéal. Le chargement en base est réparti entre :
- `etlbooking.ipynb` pour les hôtels (S3 → nettoyage → table `hotels` dans RDS).
- `meteo et cartes optimisees.ipynb` pour la météo (upload S3 → table `weather` dans RDS) **et** la visualisation.

L'idéal aurait été de centraliser toute la logique ETL dans un seul notebook et de séparer la visualisation. Cela dit, chaque notebook est autonome et le résultat final (les deux tables dans RDS, correctement remplies) est identique.

---

### Comment le score météo est-il calculé ?

Le score est calculé dans `meteo api booking v3.ipynb` avec cette formule :

```
Score = Température Moyenne − (Pluie Totale en mm × 2)
```

L'idée : un voyageur cherche un endroit chaud et sec. La température est un bonus, la pluie une pénalité.

Le brief laissait explicitement le choix de la méthode (*"you can have different opinions on what nice weather would be like 😎"*). J'aurais pu intégrer l'humidité ou le vent, mais pour un premier déploiement, la combinaison température/pluie couvre les deux critères les plus intuitifs.

---

### Pourquoi utiliser S3 comme Data Lake plutôt que charger directement dans RDS ?

L'architecture Data Lake → Data Warehouse était demandée par le brief. Mais au-delà de la consigne, j'ai réellement compris son utilité pendant le projet :

| Couche | Service | Ce que ça apporte |
|--------|---------|---------|
| **Data Lake** | Amazon S3 | Données brutes conservées telles quelles. On peut toujours revenir en arrière si le nettoyage introduit un bug. Coût quasi nul. |
| **Data Warehouse** | RDS PostgreSQL | Données nettoyées et typées. Requêtes SQL performantes. |

La séparation est particulièrement utile quand le scraping est fragile : si les cookies expirent demain, les données brutes restent sur S3.

---

### Pourquoi 699 hôtels sur 880 dans la base finale ?

Le géocodage via Nominatim ne réussit pas toujours. Certains hôtels ont des noms ambigus, des adresses en langue étrangère, ou des localisations trop vagues. Les ~180 hôtels qui manquent correspondent à ces cas.

C'est un compromis de qualité délibéré : mieux vaut 699 hôtels correctement géolocalisés que 880 avec des marqueurs mal placés sur la carte.

---

### Pourquoi les notes d'hôtels nécessitent un nettoyage Regex ?

Les notes récupérées par le scraper arrivent sous forme textuelle (*"Avec une note de 8.5"* ou *"Note : 7,2"*). Booking intègre la note dans une phrase descriptive, pas dans un champ numérique isolé.

La fonction de nettoyage extrait le premier nombre décimal trouvé dans la chaîne, puis convertit la virgule en point pour obtenir un float Python :

```python
def clean_score_v2(s):
    if pd.isna(s): return 0.0
    match = re.search(r'(\d+[.,]\d+)', str(s))
    if match:
        return float(match.group(1).replace(',', '.'))
    return 0.0
```

---

### Pourquoi `carto-positron` au lieu d'OpenStreetMap pour les cartes ?

Un problème découvert en cours de route : OpenStreetMap applique une Referrer Policy stricte qui bloque les tuiles de carte quand elles sont appelées depuis un notebook Jupyter ou un fichier HTML local. Résultat : une carte vide avec uniquement les marqueurs.

Le style `carto-positron` (fourni par CARTO) n'a pas cette restriction et offre un rendu plus épuré. C'est maintenant la recommandation standard de la communauté Plotly pour les cartes interactives.
