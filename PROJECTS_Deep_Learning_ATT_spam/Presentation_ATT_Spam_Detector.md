# Détecteur de Spam pour AT&T - Filtrage Intelligent de SMS

![AT&T Logo](https://full-stack-assets.s3.eu-west-3.amazonaws.com/M08-deep-learning/AT%26T_logo_2016.svg)

---

## 📞 Le Problème 

AT&T, multinationale américaine et leader mondial des télécommunications, fait face à une insatisfaction croissante de ses utilisateurs : **la réception constante de SMS de Spams frauduleux.**

Jusqu'à présent, les équipes d'AT&T signalaient ces messages manuellement. Ce processus est chronophage et inefficace à grande échelle.

**Objectif :**
Construire un détecteur de spam alimenté par l'Intelligence Artificielle capable de **bloquer automatiquement** les messages indésirables avant qu'ils n'atteignent les clients, en se basant uniquement sur leur contenu textuel.

---

## 📊 1. Exploration du Jeu de Données (EDA)

Je dispose d'un jeu de données de **5 572 SMS**, déjà classés en deux catégories :
- **Ham (légitime)** : 86.6% (4825)
- **Spam (indésirable)** : 13.4% (747)

**Observations clés sur le texte :**
*   **Déséquilibre :** Le dataset est débalancé (~8.5 contre 1). 
*   **Vocabulaire distinct :** Les spams abusent de mots comme *FREE*, *Call*, *Text*, *Txt*, *Mobile*, *Prize*. 
*   **Longueur des messages :** Les spams sont statistiquement beaucoup plus longs (moyenne 138 caractères) que les messages normaux (moyenne 71 caractères).

<br>

---

## 🧹 2. L'Étape Cruciale : Le Preprocessing du Texte

Avant de nourrir une intelligence artificielle, le texte humain (bruité, non structuré) doit être préparé :

1. **Nettoyage Regex** : `str.replace(r"[\W_]+", " ")`. 
   * *Principe* : Je supprime toute la ponctuation (!, ?, _, etc.).
   * *Particularité Spam* : **Je conserve explicitement les chiffres.** Les Spams contiennent presque toujours des prix de concours, des numéros courts à contacter ("Text 80082"), ou des dates limites. Supprimer les chiffres serait supprimer mon meilleur indice !
2. **Minuscules (Lowercasing)** : Uniformisation du texte.
3. **Encodage Cible** : SPAM = 1, HAM = 0.
4. **Tokenisation Keras** : Chaque mot unique de notre corpus se voit attribuer un numéro (ID). Les phrases deviennent des suites de nombres entiers !
5. **Padding (Rembourrage)** : Un réseau de neurones attend des matrices de taille fixe. J'allonge artificiellement les textes avec une suite de `zeros_` pour qu'ils mesurent tous 150 jetons de long.

<br>

---

## 🧠 3. La Modélisation : Duel de 3 Approches

Pour évaluer la véritable puissance de mon IA, je l'ai passée au crible via un processus graduel divisé en 3 modèles de complexité croissante.

### ▶ Modèle M0 : La Baseline (Regression Logistique)
*   **L'idée** : Une approche "Machine Learning Classique" basée sur des mathématiques statistiques (TF-IDF). Sans neurones.
*   **Résultat** : Un modèle ultra-rapide mais basique. Capable de bloquer certains Spams évidents, mais très fragile face au nouveau vocabulaire.
*   **Utilité** : Fixe le point de repère de performance que nos Réseaux de Neurones doivent battre !

### ▶ Modèle A : Réseau de Neurones Simple (Custom)
*   **L'idée** : Construire moi-même un "Deep Learning" léger. Les mots tokenisés passent dans une couche `Embedding` qui apprend aux algorithmes "quelles idées de fondation les mots partagent". Puis j'extrais la moyenne des idées (`GlobalAveragePooling1D`), qui est décodée par une poignée de neurones (Couches denses).
*   **Résultat** : Une précision incroyable sur l'entraînement, un excellent modèle "fait maison". Apprend extrêmement vite (presque parfait à l'époque 4). Il manque cependant de contexte humain global.

### ▶ Modèle B : Transfert Learning (L'État de l'Art)
*   **L'idée** : Utiliser un modèle de compréhension de texte de la famille *Transformers* via HuggingFace (un cousin de ChatGPT - `Sentence-Transformers`).
*   **Pourquoi ?** : Le modèle pré-entraîné a déjà "lu" tout internet (Wikipédia, Reddit...). Avant même que je lui montre les SMS de Spam, il sait *déjà* comprendre la nuance globale de l'anglais ! Mon travail n'est plus que d'entraîner le petit classifieur qui dit "ce résumé sémantique ressemble-t-il à un Spam ?"
*   **Résultat** : Une généralisation extraordinaire sans overfiting.

<br>

---

## 🏆 4. Résultats & Recommandation

| Modèle | Précision (Sûreté) | Rappel (Taux de Capture) | F1-Score (Équilibre) |
| :--- | :---: | :---: | :---: |
| **M0 : Regression Logistique** | ~96% | ~71% | ~81% |
| **Modèle A : Réseau Simple** | ~99% | ~88% | ~93% |
| **Modèle B : Transfert Learning** | **~96%** | **~96%** | **~96%** |

*(Les mesures ci-dessus s'appliquent sur la classe "Spam" du jeu de test)*

### Pourquoi le Modèle B (Tansfert Learning) est mon meilleur candidat pour la production ?

1. **Le F1-Score Majeur :** Le Modèle B est le seul à offrir un équilibre spectaculaire entre la Précision et le Rappel (96%).
2. **Le Problème du Modèle A :** Bien que son score soit beau, le modèle A a été entraîné *uniquement* sur les 5000 SMS de ma base de données. Que se passera-t-il si un Spam utilise un adverbe ou un verbe moderne qui n'existait pas dans mon jeu de données original ? Le réseau crashera. Le modèle B, lui, ayant lu Wikipédia, saura le lier sémantiquement à ses connaissances mondiales.
3. **Cas Client** : Du point de vue d'AT&T, quel est l'objectif ? Il faut éviter les "Faux Négatifs" (Laisser passer le Spam dans la boîte du client), sans pour autant commettre de "Faux Positifs" (envoyer les textos légitimes de la Famille dans les courriers indésirables !). Le modèle B fait précisément ça.

### Bilan AT&T

* ✅ Un pipeline d'analyse de texte naturel robuste et nettoyé à base d'expressions régulières.
* ✅ Une IA basée sur la toute dernière technologie d'attention (Sentence-Transformers) facile à exporter en Cloud et peu gourmande.
* ✅ L'écrasante majorité des SPAMS bloqués avant de nuire !
