# Rétrospective et Leçons Apprises — Projet Kayak

Ce document passe en revue la cohérence entre le brief initial (`01-Plan_your_trip_with_Kayak.ipynb`) et l'implémentation réelle. L'objectif n'est pas de chercher des erreurs, mais de comprendre les écarts entre la théorie et la pratique, et ce qu'ils m'ont appris.

## Vue d'ensemble

Le projet respecte l'architecture demandée : **Sources de données → Data Lake (S3) → Data Warehouse (RDS) → Visualisation (Plotly)**. Toutes les briques techniques du brief (scraping, API, ETL, stockage Cloud) sont présentes et fonctionnelles.

Les écarts relevés ne sont pas des erreurs mais des **adaptations au terrain** — le genre de choses qu'on ne peut pas prévoir tant qu'on n'a pas mis les mains dans le code.

## Analyse par fichier

### `booking avec description.ipynb` (Scraping)
- **Ce que le brief demandait :** Scraper les hôtels sur Booking.com, y compris les coordonnées GPS.
- **Ce qui s'est passé :** Le scraper récupère bien le nom, l'URL, la note, le prix et la description. Mais les coordonnées GPS sont impossibles à extraire — Booking les masque dans des widgets JavaScript dynamiques.
- **Leçon :** Un brief peut suggérer des choses qui ne sont pas réalisables telles quelles. Il faut savoir pivoter plutôt que s'obstiner.

### `geocodageafterscrap.ipynb` (Géocodage)
- **Ce que le brief prévoyait :** Rien — ce fichier est un ajout personnel.
- **Pourquoi il existe :** Pour compenser l'impossibilité de scraper les coordonnées GPS, j'ai étendu l'utilisation de Nominatim (suggéré par le brief pour les villes) aux hôtels eux-mêmes.
- **Leçon :** Un outil déjà dans le projet peut servir à résoudre un problème qu'on n'avait pas anticipé.

### `meteo api booking v3.ipynb` (Météo)
- **Cohérence avec le brief :** Très bonne. Suit les suggestions (Nominatim + OpenWeatherMap) et ajoute un score météo personnel.
- **Choix personnel :** La formule de scoring (`Temp - Pluie × 2`) — le brief laissait la liberté de la méthode.

### `etlbooking.ipynb` et `meteo et cartes optimisees.ipynb` (ETL + Visualisation)
- **Ce que le brief demandait :** Un pipeline ETL (S3 → RDS) et des cartes Plotly.
- **Ce qui n'est pas idéal :** La logique ETL est répartie sur deux fichiers. `etlbooking.ipynb` gère les hôtels, tandis que `meteo et cartes optimisees.ipynb` gère la météo ET la visualisation.
- **Ce qui aurait été mieux :** Centraliser l'ETL dans un seul fichier, et séparer la visualisation.
- **Impact réel :** Aucun. Chaque notebook est autonome, et le résultat final est correct.

## Résumé des écarts

| Type | Fichier | Description | Impact |
|:-----|:--------|:------------|:-------|
| Adaptation | `booking avec description.ipynb` | Pas de GPS scrapé (masqué par Booking). Compensé par `geocodageafterscrap.ipynb`. | Aucun — le résultat final est identique. |
| Redondance | `booking_full_data-save.csv` | Sauvegarde manuelle. Artisanal mais utile comme filet de sécurité. | Peut être supprimé. |
| Organisation | ETL sur 2 notebooks | Logique de chargement éclatée entre hôtels et météo. | Fonctionnel mais pas optimal. |

## Ce que j'en retiens

Le brief fournissait un cadre solide, et globalement le projet le respecte. Les écarts que j'ai rencontrés sont typiques du Data Engineering en conditions réelles : les données ne sont jamais aussi accessibles qu'on le pense, et l'organisation du code évolue au fil des problèmes rencontrés. L'important, c'est que le pipeline fonctionne de bout en bout et que les livrables sont tous présents.
