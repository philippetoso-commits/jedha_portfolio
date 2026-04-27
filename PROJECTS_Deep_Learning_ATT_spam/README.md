# AT&T - Détecteur de Spam Intelligent 📞

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=fff)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=fff)](#)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=000)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

La réception constante de SMS frauduleux et de spams est une source majeure d'insatisfaction pour les clients télécoms. Ce projet d'Intelligence Artificielle vise à aider **AT&T** en bloquant automatiquement les messages indésirables avant même qu'ils n'atteignent le téléphone des utilisateurs.

> ⚠️ **Note :** Ce projet a été réalisé dans le cadre de la certification en Data Science (Spécialité Deep Learning & NLP) chez JEDHA Bootcamp.

---

## 📖 Le projet en quelques mots

Au lieu de bloquer manuellement des numéros, nous utilisons le **Traitement du Langage Naturel (NLP)** et le **Deep Learning** pour analyser le contenu textuel des messages. 

Le but est d'atteindre un équilibre parfait : maximiser le taux de blocage des spams (Rappel) tout en s'assurant de ne *jamais* envoyer un message légitime de la famille dans les indésirables (Précision et faux positifs).

---

## 📊 Les sources de données

Le projet se base sur un jeu de données de **5 572 SMS** (`spam.csv`), classifiés en :
- **Ham (Légitime)** : ~86.6%
- **Spam (Indésirable)** : ~13.4%

L'exploration initiale a montré que les spams sont statistiquement plus longs et utilisent un vocabulaire d'urgence et de gratuité (*FREE, Call, Text, Prize*).

---

## ⚙️ Installation

```bash
cd "PROJECTS_Deep_Learning_ATT_spam"
pip install pandas tensorflow keras transformers jupyter
```

---

## 🚀 Les notebooks et fichiers

Le projet est documenté et structuré via les fichiers suivants :

| Fichier | Ce qu'il fait |
|---|---|
| `01-ATT_spam_detector_FR.ipynb` | Notebook complet en français (Nettoyage, Tokenization, Modèles) |
| `01-ATT_spam_detector_EN.ipynb` | Notebook d'analyse complet (Version anglaise) |
| `Presentation_ATT_Spam_Detector.md` | Synthèse métier et résultats des différents modèles |
| `discours.md` | Script d'accompagnement pour la soutenance orale |
| `FAQ.md` | Foire aux questions technique |

---

## 🏗️ Architecture et Modélisation (NLP)

Le traitement d'un texte humain en une matrice mathématique compréhensible par l'IA suit un pipeline strict :

### 1. Le Preprocessing du Texte
- **Nettoyage Regex** : Suppression de la ponctuation complexe tout en **conservant les chiffres**, qui sont des indices capitaux dans les spams (numéros courts, prix).
- **Tokenisation & Padding** : Chaque mot devient un numéro (ID) via Keras, et les phrases sont unifiées à une longueur fixe (150 jetons).

### 2. Le Duel des Modèles
Le projet explore une progression technologique :
- **Baseline (Régression Logistique)** : Approche statistique classique (TF-IDF). Rapide mais très fragile face au vocabulaire inconnu.
- **Réseau de Neurones Simple (Custom)** : Construction d'une couche `Embedding` maison couplée à un réseau Dense. Excellents résultats sur nos données, mais manque de contexte mondial.
- **L'État de l'Art (Transfert Learning - HuggingFace)** : Utilisation d'un modèle d'attention de la famille des *Transformers* (`Sentence-Transformers`). Le modèle a déjà appris à comprendre l'anglais global en lisant le web, offrant une généralisation parfaite (F1-Score de 96%).

---

## 💡 Résultats 

Le modèle basé sur le **Transfert Learning** est le grand gagnant de cette étude. Il atteint **96% de Précision et 96% de Rappel**. 
C'est le seul modèle garantissant à AT&T de bloquer efficacement les fraudes tout en préservant de manière quasi certaine l'intégrité de la messagerie personnelle de ses clients (quasi aucun faux positif).

---

## 📂 Structure du projet

```text
PROJECTS_Deep_Learning_ATT_spam/
├── spam.csv                             # Dataset brut (SMS et labels)
├── 01-ATT_spam_detector_FR.ipynb        # Notebook principal (NLP & Deep Learning)
├── 01-ATT_spam_detector_EN.ipynb        # Version anglaise du notebook
├── src/                                 # Images pour la présentation
├── Presentation_ATT_Spam_Detector.md    # Synthèse du projet
├── discours.md                          # Trame pour la présentation
├── FAQ.md                               # Anticipation Q&A
└── README.md                            # Ce fichier
```

---

## ✍️ Auteur
Projet réalisé par **Philippe Toso** dans le cadre de la formation Data Fullstack — JEDHA Bootcamp.
