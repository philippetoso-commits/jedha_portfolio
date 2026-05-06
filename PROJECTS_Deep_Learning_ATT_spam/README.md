# AT&T - Détecteur de Spam Intelligent

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=fff)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=fff)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=000)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

La réception constante de SMS frauduleux et de spams est une source majeure d'insatisfaction pour les clients télécoms. Ce projet d'Intelligence Artificielle vise à aider **AT&T** en bloquant automatiquement les messages indésirables avant même qu'ils n'atteignent le téléphone des utilisateurs.

> **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science (Spécialité Deep Learning & NLP) chez JEDHA Bootcamp.

---

## Le projet en quelques mots

Au lieu de bloquer manuellement des numéros, nous utilisons le **Traitement du Langage Naturel (NLP)** et le **Deep Learning** pour analyser le contenu textuel des messages. 

Le but est d'atteindre un équilibre parfait : maximiser le taux de blocage des spams (Rappel) tout en s'assurant de ne *jamais* envoyer un message légitime de la famille dans les indésirables (Précision et faux positifs).

---

## Les sources de données

Le projet se base sur un jeu de données de **5 572 SMS** (`spam.csv`), classifiés en :
- **Ham (Légitime)** : ~86.6%
- **Spam (Indésirable)** : ~13.4%

L'exploration initiale a montré que les spams sont statistiquement plus longs et utilisent un vocabulaire d'urgence et de gratuité (*FREE, Call, Text, Prize*).

---

## Installation

```bash
cd "PROJECTS_Deep_Learning_ATT_spam"
pip install pandas tensorflow keras transformers jupyter
```

---

## Livrables

- **Notebook principal en français** : `01-ATT_spam_detector_FR.ipynb`
- **Notebook en anglais** : `01-ATT_spam_detector_EN.ipynb`
- **Code source complet (GitHub)** : https://github.com/philippetoso-commits/jedha_portfolio
- **Lien direct dossier projet** : https://github.com/philippetoso-commits/jedha_portfolio/tree/main/PROJECTS_Deep_Learning_ATT_spam

---

## Les notebooks et fichiers

Le projet est documenté et structuré via les fichiers suivants :

| Fichier | Ce qu'il fait |
|---|---|
| `01-ATT_spam_detector_FR.ipynb` | Notebook complet en français : preprocessing, tokenisation, modèles et évaluation |
| `01-ATT_spam_detector_EN.ipynb` | Version anglaise du notebook |
| `01-AT&T_spam_detector.ipynb` | Brief original du projet |
| `spam.csv` | Dataset brut des SMS labellisés |
| `FAQ.md` | Foire aux questions technique pour préparer la soutenance |
| `src/` | Visualisations générées pour illustrer l'analyse |

---

## Architecture et Modélisation (NLP)

Le traitement d'un texte humain en une matrice mathématique compréhensible par l'IA suit un pipeline strict :

### 1. Le Preprocessing du Texte
- **Nettoyage Regex** : Suppression de la ponctuation complexe tout en **conservant les chiffres**, qui sont des indices capitaux dans les spams (numéros courts, prix).
- **Tokenisation & Padding** : Chaque mot devient un numéro (ID) via Keras, et les phrases sont unifiées à une longueur fixe (150 jetons).

### 2. Le Duel des Modèles
Le projet explore une progression technologique :
- **Baseline (Régression Logistique)** : Approche statistique classique (TF-IDF). Rapide mais très fragile face au vocabulaire inconnu.
- **Réseau de Neurones Simple (Custom)** : Construction d'une couche `Embedding` maison couplée à un réseau Dense. Il obtient le meilleur F1-score dans le notebook principal en français.
- **Transfert Learning - HuggingFace** : Utilisation de `Sentence-Transformers` pour générer des embeddings sémantiques pré-entraînés. Cette approche obtient le meilleur F1-score dans la version anglaise après réexécution.

---

## Résultats 

Les notebooks sauvegardés comparent trois approches sur le jeu de test. Les deux modèles Deep Learning obtiennent des performances proches et dépassent la baseline classique :

- **Régression Logistique** : F1-score spam ≈ 89.7%
- **Réseau de Neurones Simple** : F1-score spam ≈ 92.5% à 92.9%
- **Sentence-Transformers** : F1-score spam ≈ 92.7% à 93.8%

Le notebook principal en français retient le **Réseau de Neurones Simple**, tandis que la version anglaise réexécutée retient **Sentence-Transformers**. Cette légère variation s'explique par la part de stochasticité de l'entraînement neural. Dans les deux cas, le projet démontre bien l'apport du Deep Learning pour améliorer la détection des spams.

---

## Structure du projet

```text
PROJECTS_Deep_Learning_ATT_spam/
├── spam.csv                             # Dataset brut (SMS et labels)
├── 01-AT&T_spam_detector.ipynb          # Brief original du projet
├── 01-ATT_spam_detector_FR.ipynb        # Notebook principal (NLP & Deep Learning)
├── 01-ATT_spam_detector_EN.ipynb        # Version anglaise du notebook
├── src/                                 # Visualisations générées
├── FAQ.md                               # Anticipation Q&A
└── README.md                            # Ce fichier
```

---

## Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack - JEDHA Bootcamp.
