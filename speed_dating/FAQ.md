# Foire Aux Questions (FAQ) - Projet Speed Dating 💘

Bienvenue dans la FAQ du projet d'analyse Speed Dating. Vous trouverez ici les réponses aux questions méthodologiques récurrentes concernant le traitement du jeu de données et nos choix de modélisation.

---

### Pourquoi supprimer les vagues 6 à 9 au lieu de simplement les diviser par 10 ?
C'est une excellente question de Data Science ! *"Pourquoi jeter la donnée si on peut la sauver par une simple division ?"*

Dans l'édition originale du dataset du professeur de Columbia University, il y a un piège majeur avec ces fameuses vagues 6-9 : **ce n'est pas une simple "note sur 100"** (du genre *"je te donne 85/100"*).

Durant les vagues normales (notées sur 10), on demande aux candidats de noter chaque attribut de 1 à 10 indépendamment.  
*Exemple : Attractivité=8/10, Sincérité=7/10, Humour=6/10.*

Durant les vagues 6 à 9, les organisateurs ont complètement changé la méthodologie psychologique. Ils ont demandé aux candidats de répartir **100 points au total** (une somme nulle) entre tous les attributs du partenaire, comme un budget. Par conséquent :

Si un candidat trouve son *date* ultra-attractif et lui met 60 points sur 100, **il ne lui reste que 40 points à distribuer** pour tous les autres critères (Sincérité, Intelligence, etc.) qui vont mécaniquement tous écoper de très mauvaises notes (5 ou 10 points) !

**Si l'on se contentait de diviser par 10 :**  
Une note de "5 points" en sincérité dans la vague 7, une fois divisée par 10, donnerait **0.5/10** au partenaire ! C'est une note épouvantablement basse et fausse (une pénalité mathématique due au système de budget limité), alors que dans une vague normale sur 10, le même candidat aurait peut-être mis librement un 7/10 de sincérité tout en mettant 9/10 d'attractivité parce qu'il n'était pas bloqué par une "somme maximale de points".

L'échelle est complétement déformée par l'exercice de "répartition", d'où la très bonne pratique de **jeter ces vagues** pour ne pas polluer les analyses par rapport aux vagues classiques notées de manière isolée.

---

### Pourquoi l'individu possédant l'ID 118 est-il introuvable dans le jeu de données ?
Lorsqu'on parcourt les identifiants uniques (IID) du dataset original, on constate que l'incrémentation s'arrête à 552. Pourtant il n'y a que 551 individus testés. 
L'individu portant le numéro `118` est absent des registres. 

Il s'agit d'un phénomène très courant en Data Science sur des données humaines réelles : le candidat n'a pas pu se présenter à l'événement de Speed Dating à la dernière minute, ou bien il a demandé a posteriori la suppression de ses données personnelles du projet de recherche. Le numéro d'identification `118` ayant déjà été assigné "dans le vide", la liste passe directement du 117 au 119.

---

### Pourquoi subsiste-t-il autant de valeurs nulles (NaN) malgré le nettoyage ?
Le jeu de données original comporte **~21% de valeurs manquantes**. Dans notre script `cleaning.py`, nous n'avons intentionnellement supprimé **que les lignes où la décision de match (le "oui" ou le "non") venait à manquer**.

Pourquoi ne pas avoir supprimé toutes les lignes contenant le moindre blanc ?  
Car les organisateurs de Columbia University envoyaient des vagues de questionnaires distinctes : 
- Avant l'événement
- Le lendemain de l'événement
- 3 semaines après l'événement...

Une immensité de candidats n'ont jamais répondu au sondage "3 semaines après" ! Si nous supprimions le participant entier du dataset sous prétexte qu'il a ignoré le dernier sondage, **nous aurions perdu 80% des rendez-vous exploitables** (qui, eux, furent notés le soir même). 
C'est pour cela que dans le code des graphiques, nous effectuons une suppression locale (`dropna(subset=['variable_analysée'])`), préservant ainsi le maximum de rencontres pour les questions courantes.

---

### D'où vient la "baisse des matchs" au fil de l'événement ?
On l'observe drastiquement sur l'analyse de l'ordre (`order`) des rencontres. Plus le sujet vient à rencontrer de nouveaux partenaires dans la soirée, plus ses chances de donner un "Oui" baissent. 
Cela provient essentiellement du concept de **Fatigue Décisionnelle**. À force d'évaluer constamment l'attractivité et l'intelligence de dizaines de partenaires à la suite, le cerveau humain sature et devient plus sévère, rehaussant ses attentes ("choice overload"). Le moment idéal pour décrocher un "oui" reste donc les 5 ou 6 premiers rounds d'une soirée !

---

### L'attractivité réelle l'emporte-t-elle toujours ?
Les déclarations des participants peuvent être trompeuses (biais de désirabilité sociale). Ainsi, sur les formulaires d'intentions, les femmes classent "l'intelligence" et la "sincérité" de leur partenaire masculin bien au-dessus de son "attractivité physique".
Cependant, l'étude mathématique des "matches" réels dévoile l'inverse : chez les deux sexes, la corrélation la plus élevée (et de loin) avec une prise de décision positive pour un 2ème rencard... est **systématiquement la note qu'ils viennent de mettre en attractivité physique**. Le physique est donc la véritable variable clé de cette base de données.
