# Discours : AT&T Spam Detector (10 minutes)

*Ce document est un script guidé pour accompagner la démonstration de votre notebook pas-à-pas. Vous pouvez le lire tel quel ou vous en inspirer.*

---

## 🕒 [0:00 - 1:30] Introduction & Contexte

**Action :** Afficher le logo et la première cellule en Markdown (*Description & Objectif*).

*(Souriant)* "Bonjour à tous. Aujourd'hui, je vous présente mon projet certifiant de Deep Learning : **Le Détecteur de Spam AT&T.**
AT&T, comme beaucoup d'opérateurs, fait face à une explosion des spams par SMS. Mon objectif a donc été simple sur le papier, mais très complexe techniquement : construire une IA capable de filtrer automatiquement et précisément les messages indésirables, en lisant uniquement leur contenu textuel.

Mon approche aujourd'hui sera très concrète : je vais construire un modèle de référence simple (une Baseline probabiliste), puis monter en puissance vers les réseaux de neurones, et clôturer sur la méthode la plus avancée du marché à l'heure actuelle : le Transfert Learning de type *Transformers* (la famille d'algorithmes qui a donné naissance à ChatGPT). L'idée est de montrer qu'un modèle très complexe n'est qu'un outil... et qu'il faut pouvoir le justifier par rapport à du code basique."

---

## 🕒 [1:30 - 3:00] Exploration des Données (EDA)

**Action :** Faire dérouler les cellules de "Data Loading" à "Word Cloud" inclus.

"Tout projet de Machine Learning commence par explorer et comprendre ses données (L'Exploratory Data Analysis). Ici, j'ai utilisé un jeu de données public (*Spam.csv*) de 5 572 SMS.
Comme vous le voyez sur ce premier graphique, **les données sont très déséquilibrées (Imbalanced)** : environ 86.6% de 'Ham' (Messages Légitimes de vrais gens) contre à peine 13.4% de 'Spam' (les arnaques). J'ai dû surveiller cela de très près car c'est un piège mortel pour évaluer une métrique comme l'Accuracy pure.

Sur le graphique de *Message Length*, on s'aperçoit d'une tendance forte : un SMS spam moyen dépasse souvent les 130 caractères, contre 70 pour un humain. C'est normal : le but d'une arnaque est de faire peur, ou de décrire longuement une fausse récompense avec un lien à cliquer.
Enfin, le nuage de mots me confirme l'évidence : là où les gens n'emploient que des expressions courtes pour discuter avec leur famille (Ok, Come, Now, Ur), les spams regorgent de mots clés récurrents comme 'FREE', 'Txt', 'Call', 'Mobile' ou 'Prize'."

---

## 🕒 [3:00 - 5:00] Text Preprocessing : Le Nettoyage et la Tokenisation

**Action :** Afficher la cellule de nettoyage du texte (Section 3).

"Avant de donner ces SMS à mes modèles, je prends un instant pour parler du **Text Preprocessing**, une étape critique. L'ordinateur ne sait pas lire du texte ; il ne lit que des chiffres.

1. La toute première chose que j'ai faite, c'est utiliser une expression régulière (Regex) pour enlever la ponctuation : `str.replace`.
**Attention cependant !** Contrairement à l'analyse de sentiments classique, je ne retire **pas** les chiffres. Pourquoi ? Un SMS de Spam dépend quasi-exclusivement de chiffres pour piéger sa victime : un numéro court américain à rappeler, un prix en $, une date limite. Les effacer, c'est retirer l'indicateur principal du modèle !

2. Une fois cette règle définie, j'utilise un `Tokenizer` Keras qui va transformer tout mon vocabulaire de textes réels en listes de nombres entiers (Tokens).
3. Enfin, je termine avec le **Padding** (`pad_sequences`). Les réseaux de neurones requièrent des tenseurs d'entrée (input shape) parfaitement alignés. J'allonge donc mathématiquement tous les plus petits SMS à 150 jetons en les blindant de zéros, pour que l'architecture soit capable de traiter des lots constants (batches)."

---

## 🕒 [5:00 - 6:30] Première passe : Baseline & Modèle A

**Action :** Exécuter la 'Baseline M0' et le 'Modèle A', pointer sur le classification_report. 

"Passons à la Modélisation. Ma **Baseline M0** expérimentale utilise de la Régression Logistique Classique (scikit-learn).  Ce modèle ultra-vieillissant ne s'entraîne que quelques secondes mais obtient déjà une précision impressionnante d'environ 93%. *Cependant*, quand on regarde le *Rappel* (Recall) pour la classe Spam (la classe 1), le modèle est médiocre (autour de 70%). C'est catastrophique pour une entreprise comme AT&T : ça veut dire que près d'un tiers des arnaques passe entre les mailles du filet. Les statistiques sémantiques seules atteignent ici leur plafond de verre !

J'ai donc bâti le **Modèle A** : un réseau de neurones profond (Deep Learning). Je l'initialise avec une couche *Embedding* qui va décoder spatialement le vocabulaire et extraire les relations de contexte ("Ah ! FREE et WIN ont une distance sémantique proche en maths !") avant de l'aplatir et de le passer dans des neurones classiques.
Sur mes courbes d'entraînement (`accuracy`/`loss`), je m'aperçois que les résultats bondissent immédiatement ! L'apprentissage stagne vers la 4ème époque (grâce à l'EarlyStopping qui empêche le surapprentissage) avec un rappel avoisinant les 88%."

---

## 🕒 [6:30 - 8:30] L'artillerie Lourde : Transfer Learning

**Action :** Afficher et lancer le Modèle C (Sentence-Transformers).

"Le problème avec le Modèle A... c'est que son monde se limite aux 5000 messages que je viens de lui donner. Si demain je lui écris un SPAM en parlant de 'Bitcoin', il ignorera le mot car le jeton (Token) n'est pas dans son unique dictionnaire. Ça cassera tout le modèle une fois mis en production sur de *vrais* téléphones.

L'industrie s'oriente donc massivement vers la 3ème approche : Le **Transfer Learning** (Modèle B dans mon rapport final).
J'ai importé un modèle appelé `Sentence-Transformers` (un cousin de la lignée BERT/Huggingface). Ce modèle pré-entraîné par des milliards de mots a lu tout wikipédia, tous les sites forums etc. À la place de simplement donner un "numéro" à chaque mot, il "lit" mon message, injecte sa vaste connaissance grammaticale du monde réel, et me renvoie par magie un **Vecteur de 384 dimensions denses** hyper expressives. 

Ici, je me suis contenté de brancher quelques neurones stupides Denses derrière (classifieurs) juste pour déterminer : 'Ce vecteur représente-t-il un trait de Spam ?'. Regardez les résultats en bas : l'entraînement demande bien moins d'efforts car le sens du texte est déjà compris par l'IA en amont."

---

## 🕒 [8:30 - 10:00] Conclusion & Analyse Qualitatifs 

**Action :** Descendre sur le tableau final et surtout afficher les Matrice de Confusions et les Pires/Bons Exemples.

"Il est l'heure de conclure. Je vous présente mon classement final récapitulatif.
1. J'ai mesuré **La Précision** (ne jamais qualifier un bon message de spam et froisser un client de la perte du SMS).
2. J'ai évalué le **Rappel** (être certain de bloquer n'importe quel Spam dangereux sans exception).
3. Le **F1-Score** représente la moyenne des deux et atteste l'équilibre d'un modèle robuste.

Ici, aucune hésitation en lecture face aux matrices de confusion : **Mon modèle B gagnant (avec Transfer Learning et Embeddings Mondiaux) s'impose haut la main avec un F1-Score supérieur à 96%.**

Mais ce qui est fascinant, ce n'est pas le score, ce sont les **erreurs de prédiction** générées par le modèle, tout en bas du notebook. Elles prouvent que le modèle n'est pas stupide, les textes sont réellement trompeurs !

Regardez ce **Faux Positif** (un vrai SMS bloqué à tort) :
> *v nice off 2 sheffield tom 2 air my opinions on categories 2 b used 2 measure...*
Etant donné que les Spams abusent généralement des suites de chiffres, le modèle a paniqué en voyant le chiffre '2' répété partout pour remplacer le mot 'to' (*off 2, tom 2*). Il a pris ça pour une machine !

Regardez ce **Faux Négatif** terrifiant (un spam qui est passé) :
> *sorry i missed your call let s talk when you have the time i m on 07090201529...*
C'est du Phishing pur. Le Spammeur se fait passer pour un ami réclamant un rappel. L'IA trouve le ton très chaleureux et humain, elle ignore donc l'arnaque du numéro surtaxé à la fin. 

L'IA n'est pas magique, elle fait des statistiques temporelles. C'est pour ça qu'en production chez AT&T, ce Modèle B interceptera plus de 96% des spams de manière invisible, mais il faudra toujours le coupler à une sécurité classique (comme une liste noire de numéros connus de la police) pour pallier ces failles psychologiques.

Merci d'avoir été mon audience, je suis prêt pour toutes vos questions !"
