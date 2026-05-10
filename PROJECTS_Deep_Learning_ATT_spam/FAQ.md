# FAQ - Présentation AT&T Spam Detector

Cette FAQ regroupe les questions probables de l'audience ou du jury lors de la présentation, avec les réponses attendues pour justify les choix techniques du notebook.

---

### Q0 : D'où proviennent les données et quelle est la taille du dataset ?

**Réponse attendue :**
Il s'agit d'un corpus public bien connu en Data Science et en NLP, souvent appelé le "SMS Spam Collection Dataset".
Le dataset a été téléchargé directement depuis le bucket AWS fourni pour le projet :
[https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv)

Il contient précisément **5 572 SMS** originaux en anglais. L'intérêt majeur de ce jeu de données est son approche 100% linguistique : il n'y a aucune métadonnée cachée (pas de numéro d'expéditeur, pas d'horodatage). L'IA n'a donc que le texte pur pour juger s'il s'agit d'une arnaque.

---

### Q1 : Pourquoi avoir gardé les chiffres dans votre Regex de nettoyage (`str.replace(r"[\W_]+", " ")`) au lieu de tout enlever par défaut ? 

**Réponse attendue :** 
Contrairement à un article de blog ou un livre, les SMS de Spam sont extrêmement dépendants des chiffres pour attirer leurs victimes. Supprimer les chiffres reviendrait à effacer le meilleur indicateur de Spam ! Un Spam typique dit par exemple : *"Call 08001234 to win 1000$ now, text 82828"*. Si l'on retire les numéros, il ne reste que *"Call to win now text"*, ce qui perd tout le contexte d'arnaque (montant, numéro surtaxé). C'est pour cela que la Regex supprime *uniquement* la ponctuation et préserve consciemment l'alphanumérique.

### Q2 : Pourquoi y a-t-il autant de différences entre l'Accuracy (Exactitude) et le F1-Score sur le modèle de test (Régression Logistique) ?

**Réponse attendue :**
Le jeu de données est très déséquilibré (imbalanced) : environ 86% des messages sont des SMS normaux (Ham) et 14% sont du Spam.
Si un algorithme "bête" devine que *tout* est normal sans même lire les textes, il aura automatiquement **86% d'Accuracy** ! C'est pour ça que la métrique (Accuracy) est **trompeuse**. 
Le F1-Score (plus précisément sur la classe Spam "1") calcule la moyenne harmonique entre la *Précision* (quand je dis Spam, est-ce vraiment un Spam ?) et le *Rappel* (ai-je trouvé tous les Spams existants ?). Le F1-Score est donc le vrai reflet de la performance sur ce dataset.

### Q3 : La Régression Logistique (Baseline) a un très bon score de *Précision* (presque pas de Faux Positifs). Pourquoi s'embêter avec du Deep Learning coûteux ?

**Réponse attendue :** 
C'est tout le paradoxe de la Régression Logistique ici : elle est **précise**, mais son **rappel** (Recall) reste plus faible que celui des meilleurs modèles de Deep Learning. Cela veut dire qu'elle fait peu d'erreurs quand elle filtre un Spam (très peu de SMS légitimes bloqués à tort), **MAIS** elle laisse passer davantage de vrais Spams dans la boîte de réception du client. Du point de vue d'AT&T, améliorer le rappel est essentiel, d'où le passage au Deep Learning. Sur les notebooks sauvegardés, les modèles neuronaux obtiennent les meilleurs F1-scores sur la classe spam.

### Q4 : Quelle est l'utilité réelle du Padding (`pad_sequences`) ? Ne perd-on pas l'information de la "longueur originelle" (EDA) qui prouvait que les Spams sont plus longs ?

**Réponse attendue :**
Le Padding est une contrainte mathématique. Un réseau de neurones standard (Dense, CNN) s'attend à recevoir en entrée des matrices/tenseurs de tailles strictes et identiques (ex: `shape=(None, 150)`). On ne peut pas lui donner une phrase de 5 mots suivie d'une phrase de 100 mots. L'ajout de 0 à la fin (`padding="post"`) permet d'homogénéiser toutes les lignes. 
L'information sur la longueur originelle n'est pas "perdue" avec des architectures comme les Embeddings : la densité des mots vs le nombre de zéros restants donne quand même un signal clair algorithmiquement !

### Q5 : Pourquoi avoir testé le Transfert Learning HuggingFace/Sentence-Transformers s'il n'est pas le meilleur modèle final ?

**Réponse attendue :**
Le Transfert Learning est pertinent parce qu'un modèle pré-entraîné comme Sentence-Transformers possède déjà une représentation sémantique riche de l'anglais, apprise sur de très grands corpus. Il peut donc mieux comprendre du vocabulaire absent du dataset initial. Cependant, le meilleur modèle en production n'est pas forcément le plus sophistiqué : on choisit celui qui obtient les meilleurs résultats mesurés sur le jeu de test. Dans ce projet, le réseau simple et Sentence-Transformers sont très proches ; le notebook français retient le réseau simple, tandis que la version anglaise réexécutée retient Sentence-Transformers. Cette variation reste cohérente avec un entraînement neural légèrement stochastique.

### Q6 : En analysant vos pires erreurs de prédiction à la fin du notebook, le modèle omet des spams évidents (Faux Négatifs). N'est-ce pas dangereux ?

**Réponse attendue :**
Si, bien sûr, mais cela montre surtout les limites psychologiques du Machine Learning pur. Prenez l'exemple du spammeur qui écrit : *"sorry i missed your call let s talk when you have the time i m on 07090201529"*. Le modèle se fait avoir car lexicalement (la politesse, le ton intime), c'est un message parfait d'humain à humain (Ham). L'IA voit bien les chiffres à la fin, mais ne peut pas savoir que ce numéro précis est surtaxé ou frauduleux. 
C'est pour cela qu'en architecture de production chez un géant comme AT&T, ce modèle d'IA ne tourne jamais de façon isolée. Il ferait office de "Filtre Sémantique Global", mais serait branché en série avec un "Filtre Déterministe" classique (qui bloque mathématiquement tous les numéros blacklistés par la police).
