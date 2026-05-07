# Architecture du système ALPR Parking

> Document de référence pour la soutenance — Certification Data Fullstack JEDHA Bootcamp
> Auteur : Philippe Toso

**Liens** : [Démo Hugging Face](https://huggingface.co/spaces/philippetos/projetplaques) · [Code GitHub](https://github.com/philippetoso-commits/jedha_portfolio/tree/main/projet%20plaque)

![Architecture ALPR Parking](./architecture_hero.png)

---

## Table des matières

1. [Pitch en 30 secondes](#pitch-en-30-secondes)
2. [Vue d'ensemble — Architecture en 4 couches](#vue-densemble--architecture-en-4-couches)
3. [Flux de données runtime](#flux-de-données-runtime)
4. [Modèle C4 — niveau Container](#modèle-c4--niveau-container)
5. [Modèle de données — Base SQLite](#modèle-de-données--base-sqlite)
6. [Stack technique et arbitrages](#stack-technique-et-arbitrages)
7. [Boucle d'apprentissage actif (Active Learning)](#boucle-dapprentissage-actif-active-learning)
8. [Décisions d'architecture (ADR light)](#décisions-darchitecture-adr-light)
9. [Points forts à mettre en avant](#points-forts-à-mettre-en-avant)

---

## Pitch en 30 secondes

**ALPR Parking** est une application web complète de **reconnaissance automatique de plaques d'immatriculation** pour le contrôle d'accès d'un parking résidentiel.

Le système combine :

- **Deux modèles de Computer Vision** (détection plaque + détection marque) en *transfer learning* sur YOLOv8
- **Un modèle d'OCR** spécialisé (`fast-plate-ocr`, 220 000 plaques d'entraînement)
- **Une base de données** des résidents autorisés (SQLite)
- **Une interface web** complète (Gradio) avec administration en temps réel
- **Une boucle d'apprentissage actif** branchée sur Hugging Face Hub

> Déployé en conteneur Docker sur Hugging Face Spaces.

---

## Vue d'ensemble — Architecture en 4 couches

L'application suit une architecture en couches strictes, où chaque module a une responsabilité unique.

```mermaid
graph TB
    subgraph UI["COUCHE PRESENTATION - app.py (Gradio) "]
        T1["Image "]
        T2["Video / GIF "]
        T3["Administration "]
        T4["Historique "]
        T5["Incoherences "]
        T6["Reglages "]
    end

    subgraph BIZ["COUCHE METIER - utils/"]
        P["ALPRPipeline<br/>(pipeline.py) "]
        AC["AccessController<br/>(access_control.py) "]
        VP["VideoProcessor<br/>(video_processor.py) "]
        VZ["Visualizer<br/>(visualizer.py) "]
        ST["Storage HF<br/>(storage.py) "]
        EG["ErrorGallery<br/>(error_gallery.py) "]
        CA["VideoCache<br/>(cache.py) "]
    end

    subgraph AI["COUCHE MODELES IA "]
        Y1["YOLOv8 Plaques<br/>(best.pt) "]
        SY["SimpleYOLO maison<br/>(modelemaison.pt) "]
        Y2["YOLOv8 Marque<br/>(my_best_model.pt) "]
        OCR["fast-plate-ocr<br/>mobile-vit-v2 "]
    end

    subgraph DATA["COUCHE PERSISTANCE "]
        DB[("SQLite<br/>alpr.db ")]
        FS["Fichiers<br/>models/*.pt "]
        HF[("Hugging Face<br/>Dataset ")]
        OUT["outputs/<br/>videos et cache "]
    end

    UI --> BIZ
    P --> Y1
    P --> SY
    P --> Y2
    P --> OCR
    AC --> DB
    VP --> P
    VP --> CA
    CA --> OUT
    ST --> HF
    P --> FS

    classDef ui fill:#3b82f6,color:#fff,stroke:#1e40af
    classDef biz fill:#8b5cf6,color:#fff,stroke:#5b21b6
    classDef ai fill:#ec4899,color:#fff,stroke:#9d174d
    classDef data fill:#10b981,color:#fff,stroke:#065f46

    class T1,T2,T3,T4,T5,T6 ui
    class P,AC,VP,VZ,ST,EG,CA biz
    class Y1,SY,Y2,OCR ai
    class DB,FS,HF,OUT data
```

**Lecture du schéma** : la couche présentation ne connaît que la couche métier. La couche métier orchestre les modèles IA et la persistance. Aucun couplage transverse, ce qui rend l'application maintenable et testable.

---

## Flux de données runtime

Trace exacte d'une requête utilisateur, de l'upload à la décision d'ouverture de barrière.

```mermaid
flowchart LR
    A["Upload<br/>image / video "] --> B["Estimation<br/>conditions<br/>(luminosite, flou) "]
    B --> C["YOLOv8<br/>Detection plaque<br/>(bbox + score) "]
    C --> D["ROI<br/>Crop niveaux de gris "]
    D --> E["fast-plate-ocr<br/>Lecture texte "]
    E --> F["YOLOv8 Marque<br/>Detection vehicule "]
    F --> G{"AccessController<br/>normalize +<br/>lookup SQLite "}
    G -->|"Plaque dans<br/>l'allowlist "| H["ACCES<br/>AUTORISE "]
    G -->|"Inconnue ou<br/>bloquee "| I["ACCES<br/>REFUSE "]
    F -.->|"Marque detectee<br/>differente DB "| J["INCOHERENCE<br/>table inconsistencies "]
    H --> K["Visualizer<br/>Banniere et annotations "]
    I --> K
    J --> K
    K --> L["Logs SQLite<br/>(audit trail) "]
    J -.->|"Correction admin "| M["storage.py<br/>Hugging Face Dataset "]

    style A fill:#dbeafe,stroke:#2563eb,color:#1e40af
    style H fill:#dcfce7,stroke:#16a34a,color:#14532d
    style I fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    style J fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    style M fill:#fef3c7,stroke:#d97706,color:#78350f
```

**Particularité** : le système ne se contente pas d'autoriser ou de refuser. Il vérifie aussi la **cohérence carte grise** (la marque détectée correspond-elle à celle déclarée pour cette plaque ?). Toute anomalie alimente la boucle d'apprentissage actif.

---

## Modèle C4 — niveau Container

Vision « architecte logiciel » : qui parle à qui, à travers quels protocoles.

```mermaid
graph TB
    user("Agent parking ")
    admin("Administrateur ")

    subgraph alpr["SYSTEME ALPR PARKING "]
        web["Interface Web<br/>Gradio - Python 3.10<br/>Port 7860 "]
        pipe["Pipeline IA<br/>YOLOv8 - OCR - PyTorch "]
        db[("Base de donnees<br/>SQLite - alpr.db ")]
        cache[("Cache video<br/>SHA256 - outputs/cache ")]
    end

    hf("Hugging Face Hub<br/>Dataset - Active Learning ")
    docker("Hugging Face Spaces<br/>Docker - CPU Free Tier ")

    user -->|"Upload image/video<br/>HTTPS "| web
    admin -->|"Gere residents<br/>HTTPS "| web
    web -->|"Appelle "| pipe
    web -->|"CRUD residents/logs "| db
    pipe -->|"Lit/ecrit logs "| db
    pipe -->|"Stocke videos "| cache
    pipe -.->|"Push corrections<br/>API HF "| hf
    docker -.->|"Heberge "| alpr

    classDef person fill:#1e40af,color:#fff,stroke:#1e3a8a,stroke-width:2px
    classDef container fill:#7c3aed,color:#fff,stroke:#5b21b6,stroke-width:2px
    classDef storage fill:#059669,color:#fff,stroke:#047857,stroke-width:2px
    classDef external fill:#ea580c,color:#fff,stroke:#c2410c,stroke-width:2px

    class user,admin person
    class web,pipe container
    class db,cache storage
    class hf,docker external
```

---

## Modèle de données — Base SQLite

Quatre tables couvrent la totalité du domaine métier.

```mermaid
erDiagram
    RESIDENTS {
        int id PK
        string plaque UK
        string marque
        string nom
        string prenom
        int age
        string telephone
        string adresse
        string ville
        string code_postal
        string abonnement
        string acces
    }

    LOGS {
        int id PK
        string plaque
        string timestamp
        string resultat
        string normalized_plate
    }

    INCONSISTENCIES {
        int id PK
        string plaque FK
        string marque_detectee
        string marque_attendue
        string nom_resident
        string timestamp
        string status
        string corrected_brand
        string image_path
        string box_json
    }

    SETTINGS {
        string key PK
        string value
    }

    RESIDENTS ||--o{ LOGS : genere
    RESIDENTS ||--o{ INCONSISTENCIES : declenche
```

**Index de performance** :

- `idx_logs_plaque` sur `logs(plaque)` permet la recherche rapide d'historique
- `idx_logs_timestamp` sur `logs(timestamp DESC)` permet le tri des logs récents

---

## Stack technique et arbitrages

| Couche | Technologie retenue | Justification |
|---|---|---|
| **Détection plaques** | Ultralytics YOLOv8n + Transfer Learning | Léger (~6 Mo), CPU-compatible, état de l'art, fine-tuning rapide (100 époques) |
| **Détection plaques (alt.)** | SimpleYOLO maison (PyTorch) | Démonstration de maîtrise : architecture grid 13x13 implémentée *from scratch* |
| **OCR** | fast-plate-ocr (`global-plates-mobile-vit-v2`) | Pré-entraîné sur 220 k plaques, 40+ pays, faible latence |
| **Détection marque** | YOLOv8 fine-tuné (`my_best_model.pt`) | Cohérence avec la stack de détection principale |
| **Backend BDD** | SQLite | Zero-config, fichier unique, suffisant pour la volumétrie cible (< 10 k résidents) |
| **Interface** | Gradio 4 | Démo IA native, itération rapide, support Spaces officiel |
| **Traitement vidéo** | OpenCV + ffmpeg | Codec adaptatif (vp80/avc1/mp4v) + ré-encodage H.264 pour compatibilité navigateur |
| **Active Learning** | huggingface_hub API | Standard de fait pour partage de datasets ML |
| **Conteneurisation** | Docker (`python:3.10-slim`, user 1000) | Sécurité (non-root) + reproductibilité bit-à-bit |
| **Hébergement** | Hugging Face Spaces (Docker SDK) | Gratuit, intégré à l'écosystème ML, déploiement Git push |

### Pourquoi *ne pas* avoir choisi…

- **PostgreSQL** : surdimensionné pour un parking résidentiel ; SQLite suffit largement
- **Streamlit** : moins adapté aux applis IA *interactives* avec uploads multiples (Gradio est natif)
- **EasyOCR / Tesseract** : moins précis sur plaques que `fast-plate-ocr` qui est *spécialisé*
- **Docker Compose / Kubernetes** : un seul container, l'orchestration apporterait de la complexité sans bénéfice

---

## Boucle d'apprentissage actif (Active Learning)

C'est l'élément architectural différenciant du projet : le système **apprend de ses erreurs** sans intervention humaine du data scientist.

```mermaid
flowchart LR
    A["1. Detection<br/>incoherence<br/>(marque differente DB) "] --> B["2. Stockage<br/>table inconsistencies<br/>+ image disque "]
    B --> C["3. Admin corrige<br/>la vraie marque<br/>(UI Gradio) "]
    C --> D["4. storage.py<br/>upload_to_dataset "]
    D --> E["5. Hugging Face<br/>Hub Dataset<br/>images/ - labels/ - meta/"]
    E -.->|"Reentrainement<br/>periodique "| F["6. Nouveau modele<br/>YOLOv8 marque "]
    F -.->|"Hot reload via<br/>reload_model "| A

    style A fill:#fee2e2,stroke:#dc2626
    style E fill:#fef3c7,stroke:#d97706
    style F fill:#dcfce7,stroke:#16a34a
```

**Bénéfices** :

- Le dataset s'enrichit automatiquement de cas réels challengeants
- Pas de fuite de données : seules les incohérences sont remontées (RGPD-friendly)
- Découplage temporel : entraînement asynchrone sans bloquer la production

---

## Décisions d'architecture (ADR light)

### ADR-001 : Pourquoi un dossier `utils/` plutôt qu'un package installable ?

**Contexte** : déploiement sur Hugging Face Spaces avec `app.py` à la racine.
**Décision** : code métier dans `utils/` importé en chemins relatifs.
**Conséquence** : déploiement simplifié (un seul `Dockerfile`), au prix d'une réutilisation cross-projet limitée.

### ADR-002 : Hot-reload des modèles plutôt que redémarrage

**Contexte** : Spaces gratuit limité en RAM (16 Go) ; redémarrer = ~30 s d'indisponibilité.
**Décision** : méthode `reload_model()` qui charge le nouveau modèle **avant** de libérer l'ancien (`gc.collect()`).
**Conséquence** : pic transitoire de RAM mais zéro downtime utilisateur.

### ADR-003 : Cache vidéo basé sur SHA256

**Contexte** : retraiter une vidéo de 30 s prend 60 à 120 s sur CPU.
**Décision** : cache `outputs/cache/` indexé par hash (10 premiers Mo) + paramètres de traitement.
**Conséquence** : démos répétées instantanées, économie CPU significative.

### ADR-004 : Normalisation des plaques avant comparaison

**Contexte** : OCR retourne `AB-123-CD` ou `AB 123 CD` selon la qualité.
**Décision** : `re.sub(r'[^A-Z0-9]', '', text.upper())` avant lookup allowlist.
**Conséquence** : tolérance native aux variations de format ; pas de faux refus pour cause de séparateur.

---

## Points forts à mettre en avant

Pour la soutenance, voici les arguments architecturaux à dérouler dans cet ordre :

1. **Application web complète, pas un notebook** : preuve de bout en bout
2. **Deux modèles YOLO + un OCR orchestrés** : maîtrise multi-modèles
3. **Modèle from-scratch** (SimpleYOLO) : compréhension de ce qui se passe sous le capot
4. **Architecture 4 couches respectée** : code maintenable et testable
5. **Boucle d'Active Learning** : le système s'améliore tout seul
6. **Déploiement reproductible** (Docker + HF Spaces) : pas un POC bricolé
7. **Conformité RGPD** : minimisation des données, pas d'identification de personnes
8. **Optimisations production** : hot-reload, cache, codec adaptatif, index SQL

---

## Annexes

- Documentation fonctionnelle complète : [`projetplaquetransfert/docs/DOCUMENTATION_COMPLETE.md`](./projetplaquetransfert/docs/DOCUMENTATION_COMPLETE.md)
- Détail du modèle from-scratch : [`projetplaquetransfert/docs/ARCHITECTURE_MODELE_MAISON.md`](./projetplaquetransfert/docs/ARCHITECTURE_MODELE_MAISON.md)
- Nouvelles fonctionnalités v2 : [`projetplaquetransfert/docs/NOUVELLES_FONCTIONNALITES.md`](./projetplaquetransfert/docs/NOUVELLES_FONCTIONNALITES.md)
- Administration SQL : [`projetplaquetransfert/docs/SQL_ADMINISTRATION.md`](./projetplaquetransfert/docs/SQL_ADMINISTRATION.md)
- Visuel de présentation (slide d'ouverture) : [`architecture_hero.png`](./architecture_hero.png)
- Schéma technique des 4 couches (HD 2400 px) : [`architecture_technique.png`](./architecture_technique.png)
- Schéma du flux runtime (HD 2400 px) : [`architecture_flux.png`](./architecture_flux.png)
- Code source de l'application : [`projetplaquetransfert/`](./projetplaquetransfert/)

---

*Document généré pour la soutenance de certification — Data Fullstack JEDHA Bootcamp*
