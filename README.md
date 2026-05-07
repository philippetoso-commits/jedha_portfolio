# Portfolio Data — Certification CDSD (JEDHA)

Ce dépôt regroupe les projets réalisés dans le cadre du parcours **Fullstack Data Science** et de la préparation à la **Certification des Science des Données (CDSD)**. Chaque dossier contient son propre `README` (contexte, données, méthodes, résultats).

Rappel consigne : 
> **Présentation à l’examen** : selon le bloc, vous **déposez** plusieurs livrables mais vous **en présentez qu’un seul** au choix (sauf lorsque le règlement impose un projet unique). Les temps d’oral sont en général **5 minutes de présentation + 5 minutes de questions**.

---

## Cartographie des blocs et des projets (dans ce dépôt)

Les intitulés correspondent au référentiel CDSD ; les liens pointent vers les répertoires du présent repository.

| Bloc | Thème (FR) | Theme (EN) | Projet(s) dans ce repo | Dossier |
|:---:|:---|:---|:---|:---|
| **1** | Construction et alimentation d’une infrastructure de gestion de données | Build & Manage a Data Infrastructure | Data Collection & Management — **Kayak** | [`projet kayak/`](./projet%20kayak/) |
| **2** | Analyse exploratoire, descriptive et inférentielle | Exploratory Data Analysis | **Tinder / Speed Dating** (EDA)<br>**Steam** (Big Data) | [`speed_dating/`](./speed_dating/)<br>[`projet steam/`](./projet%20steam/) |
| **3** | Analyse prédictive — données **structurées** (Machine Learning) | Machine Learning — structured data | **Walmart Sales**<br>**Conversion rate challenge**<br>**Uber Pickups**<br>**The North Face e-commerce** | [`Wallmart/`](./Wallmart/)<br>[`conversion rate challenge/`](./conversion%20rate%20challenge/)<br>[`Uber_Pickups/`](./Uber_Pickups/)<br>[`The North Face ecommerce/`](./The%20North%20Face%20ecommerce/) |
| **4** | Analyse prédictive — données **non structurées** (Deep Learning) | Deep Learning | **Projet AT&T** (spam / texte) | [`PROJECTS_Deep_Learning_ATT_spam/`](./PROJECTS_Deep_Learning_ATT_spam/) |
| **5** | Industrialisation, déploiement et automatisation | Deployment | **Getaround** (API, suivi ML, tableau de bord) | [`Analyse Getaround/`](./Analyse%20Getaround/) |
| **6** | Direction de projet data | Lead a Data Project | **Projet final — ALPR Parking** (vision, livrable type produit démo déployée) | [`projet plaque/`](./projet%20plaque/) |

Projets hors parcours FDS présents dans d’autres versions du programme (Snowflake, Netflix, etc.) ne sont pas listés ici faute de copie dans ce dépôt.

---

## Aperçu rapide par compétence

| Domaine | Exemple de réalisation dans le repo |
|---|---|
| Ingénierie des données | Pipeline Kayak — collecte (Scrapy), qualité des données, restitution géographique / hôtellerie |
| EDA / statistiques | Speed Dating ; analyses à grande échelle Steam |
| ML tabulaire | Prévisions et modélisation Walmart, Conversion, Uber, North Face |
| Deep Learning NLP | Détection/classification spam type AT&T |
| MLOps / déploiement | Getaround — expérimentations, exposition de modèle, monitoring type dashboard |
| Computer Vision & produit | ALPR — détection / OCR, base locale, interface Gradio, conteneurisation |

---

## Liens utiles 

| Ressource | URL |
|---|---|
| Dépôt GitHub (racine) | `https://github.com/philippetoso-commits/jedha_portfolio` |
| Démo ALPR (Hugging Face Space) | `https://huggingface.co/spaces/philippetos/projetplaques` |

Remplacez ou complétez la section « Liens utiles » (LinkedIn, CV, autres Spaces) selon votre profil public.

---

## Convention des dossiers

- Chaque projet est **autonome** (notebooks, scripts, `requirements` ou équivalent selon le livrable).
- Les **environnements virtuels** (`venv/`, `.venv/`) et les **gros jeux de données** ne sont en principe pas versionnés : consulter le `README` du projet concerné pour la reproduction.

---

*Formation : JEDHA Bootcamp — Certification Data Fullstack / CDSD.*
