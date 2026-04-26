# Projet Kayak ✈️ — Plan Your Trip

Quelles sont les meilleures destinations en France pour partir en vacances ? Ce projet construit un pipeline de données complet — du scraping web au Data Warehouse cloud — pour répondre à cette question avec des données réelles.

## 1. Le contexte

Kayak (filiale de Booking Holdings, +300M$ de CA annuel) constate que **70 % de ses utilisateurs voudraient plus d'informations fiables sur leurs destinations** avant de réserver. L'équipe marketing souhaite créer une application de recommandation basée sur des données factuelles.

Le brief du projet (`01-Plan_your_trip_with_Kayak.ipynb`) fournissait déjà un cadre assez détaillé : les 35 villes à analyser, les APIs à utiliser (Nominatim, OpenWeatherMap), la méthode de collecte (scraping Booking.com), et l'architecture Cloud cible (S3 → RDS). Mon travail a consisté à implémenter tout ça et à m'adapter quand les choses ne se passaient pas comme prévu.

```python
# Les 35 villes du périmètre (fournies par le brief)
cities = [
    "Mont Saint Michel", "St Malo", "Bayeux", "Le Havre", "Rouen",
    "Paris", "Amiens", "Lille", "Strasbourg", "Chateau du Haut Koenigsbourg",
    "Colmar", "Eguisheim", "Besancon", "Dijon", "Annecy",
    "Grenoble", "Lyon", "Gorges du Verdon", "Bormes les Mimosas", "Cassis",
    "Marseille", "Aix en Provence", "Avignon", "Uzes", "Nimes",
    "Aigues Mortes", "Saintes Maries de la mer", "Collioure", "Carcassonne",
    "Ariege", "Toulouse", "Montauban", "Biarritz", "Bayonne", "La Rochelle"
]
```

## 2. Collecte des données météo — API OpenWeatherMap

Première étape, directement guidée par le brief : savoir quel temps il fera dans chacune des 35 villes.

- **Nominatim** (OpenStreetMap) convertit le nom d'une ville en coordonnées GPS. Gratuit, sans clé API.
- **OpenWeatherMap** renvoie les prévisions sur 5 jours par blocs de 3 heures.

```python
# Géocodage d'une ville via Nominatim (suggéré par le brief)
import requests

def get_coordinates(city):
    url = f"https://nominatim.openstreetmap.org/search?q={city},France&format=json&limit=1"
    headers = {"User-Agent": "kayak-project/1.0"}
    response = requests.get(url, headers=headers)
    data = response.json()
    return float(data[0]['lat']), float(data[0]['lon'])
```

Le **score météo** est un choix personnel — le brief laissait la liberté de la formule (*"you can have different opinions on what a nice weather would be like 😎"*) :

```
Score = Température Moyenne − (Pluie Totale en mm × 2)
```

| Rang | Ville | Temp. Moy. | Pluie | Score |
|------|-------|-----------|-------|-------|
| 1 | Bormes-les-Mimosas | 14.6°C | 0.6 mm | 13.4 |
| 2 | Cassis | 15.0°C | 2.6 mm | 9.8 |
| 3 | Marseille | 15.4°C | 3.0 mm | 9.4 |
| 4 | Aix-en-Provence | 15.0°C | 2.9 mm | 9.2 |
| 5 | La Rochelle | 11.6°C | 2.5 mm | 6.6 |

## 3. Scraping des hôtels — Booking.com

C'est ici que la théorie du brief s'est heurtée à la réalité. Le plan demandait de scraper Booking.com pour récupérer les hôtels (nom, URL, note, coordonnées GPS, description). Simple sur le papier.

**Données extraites :**
- Nom de l'hôtel, URL, note, prix, description

```python
# Extrait simplifié du spider Scrapy
import scrapy

class BookingSpider(scrapy.Spider):
    name = "booking"
    
    def parse(self, response):
        for hotel in response.css('div[data-testid="property-card"]'):
            yield {
                'nom_hotel': hotel.css('div[data-testid="title"]::text').get(),
                'url': hotel.css('a[data-testid="title-link"]::attr(href)').get(),
                'score': hotel.css('div[data-testid="review-score"]::text').get(),
                'prix': hotel.css('span[data-testid="price-and-discounted-price"]::text').get(),
                'description': hotel.css('div.abf093bdfe::text').get(),
            }
```

**Ce que le brief ne disait pas :** Booking.com utilise un système anti-bot AWS WAF très agressif. Impossible de scraper "naïvement". J'ai dû injecter des cookies de session valides extraits de mon navigateur et ajouter des délais entre les requêtes pour ne pas être banni.

Résultat : **880 hôtels scrapés** sur les 35 villes.

### Le problème des coordonnées GPS

Le brief suggérait de scraper les coordonnées directement. En réalité, Booking les masque dans des widgets Google Maps chargés dynamiquement — impossible à capturer avec un scraper statique. J'ai contourné le problème avec un **géocodage a posteriori via Nominatim** (la même API que pour les villes, mais appliquée aux adresses des hôtels).

```python
# Géocodage des hôtels — adaptation non prévue par le brief
from geopy.geocoders import Nominatim
import time

geolocator = Nominatim(user_agent="kayak-project")

def geocode_hotel(name, city):
    query = f"{name}, {city}, France"
    location = geolocator.geocode(query)
    time.sleep(1)  # Respect du rate-limit
    if location:
        return location.latitude, location.longitude
    return None, None
```

Après géocodage, **699 hôtels** sur 880 ont des coordonnées valides. Les autres avaient des adresses trop ambiguës.

## 4. Architecture Cloud — AWS

L'architecture Data Lake → Data Warehouse était demandée par le brief. Voici comment je l'ai implémentée :

```
                  ┌─────────────────────────────┐
                  │     SOURCES DE DONNÉES       │
                  │  Scrapy (Booking) + API Météo │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │     DATA LAKE — Amazon S3    │
                  │  booking_final_gps.csv       │
                  │  weather_final.csv           │
                  │  (Données brutes)            │
                  └──────────────┬───────────────┘
                                 │  ETL Pipeline
                                 ▼
                  ┌─────────────────────────────┐
                  │  DATA WAREHOUSE — Amazon RDS │
                  │  PostgreSQL (SSL)             │
                  │  ┌─────────┐  ┌────────────┐ │
                  │  │ hotels  │  │  weather   │ │
                  │  │ 699 rows│  │  35 rows   │ │
                  │  └─────────┘  └────────────┘ │
                  └──────────────┬───────────────┘
                                 │  SQL Queries
                                 ▼
                  ┌─────────────────────────────┐
                  │     VISUALISATION — Plotly   │
                  │  Top 5 Destinations          │
                  │  Top 20 Hôtels               │
                  └─────────────────────────────┘
```

### Pipeline ETL

```python
# 1. EXTRACT — Téléchargement depuis S3
obj = s3.get_object(Bucket=BUCKET_NAME, Key="booking_final_gps.csv")
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# 2. TRANSFORM — Nettoyage des notes (les notes arrivent sous forme de texte)
def clean_score_v2(s):
    """Transformer 'Avec une note de 8.5' en float 8.5"""
    if pd.isna(s): return 0.0
    match = re.search(r'(\d+[.,]\d+)', str(s))
    return float(match.group(1).replace(',', '.')) if match else 0.0

df['score'] = df['score'].apply(clean_score_v2)

# 3. LOAD — Injection dans RDS PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}?sslmode=require")
df_final.to_sql('hotels', engine, if_exists='replace', index=False, chunksize=50)
```

## 5. Résultats

Le notebook `meteo et cartes optimisees.ipynb` croise les données météo et hôtelières via des requêtes SQL, puis génère deux cartes Plotly :

- **Carte 1 — Top 5 Destinations** : les villes avec le meilleur score météo.
- **Carte 2 — Top 20 Hôtels** : les meilleurs établissements dans les zones recommandées.

*(Les cartes utilisent le style `carto-positron` au lieu d'OpenStreetMap, pour contourner un problème de Referrer Policy dans Jupyter.)*

## 6. Bilan

### Ce qui a été livré
| Livrable | Description |
|----------|-------------|
| CSV sur S3 | Données météo et hôtelières brutes |
| Table `hotels` sur RDS | 699 hôtels nettoyés et géolocalisés |
| Table `weather` sur RDS | 35 villes avec scores météo |
| 2 cartes Plotly | Top 5 destinations + Top 20 hôtels |

### Ce qui pourrait être amélioré
- **Centraliser l'ETL** : Avoir un seul notebook ETL au lieu de deux.
- **Automatiser** : Planifier via AWS Lambda ou Airflow pour des données toujours à jour.
- **Enrichir le scoring** : Intégrer l'humidité, le vent, ou un score NLP sur les descriptions d'hôtels.
- **Interface utilisateur** : Déployer un dashboard Streamlit pour filtrer par budget et température.
