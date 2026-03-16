# Script Oral – The North Face E-commerce
## Soutenance de 10 Minutes – Jedha Bootcamp
### Machine Learning Non Supervisé appliqué à un Catalogue Produits

---

> **Instructions :** Ce script est horodaté avec des repères de temps indicatifs.
> Ton : vulgarisateur mais professionnel. Imaginez que vous expliquez à un directeur marketing qui connaît le business mais pas le Machine Learning.

---

## [00:00 – 01:00] INTRODUCTION : Accrocher le jury

*[Ton : dynamique, accrocheur]*

"Bonjour à toutes et à tous.

Imaginez que vous entrez dans un magasin The North Face à Paris. Un vendeur expert vous accueille. Vous lui dites : 'Je cherche une veste pour l'escalade en haute montagne.' En 30 secondes, il vous emmène au bon rayon, vous montre 3 vestes parfaitement adaptées, et vous dit : 'D'ailleurs, les clients qui achètent cette veste prennent souvent aussi ce pantalon et cette sous-couche.'

Voilà ce que notre projet veut reproduire. Pas avec un vendeur humain — mais avec des algorithmes de Machine Learning.

The North Face vend des centaines de produits en ligne. Le problème ? Les clients se perdent, ne trouvent pas ce qui leur correspond, et quittent le site sans acheter. Notre mission : utiliser le contenu des descriptions produits pour créer automatiquement cette intelligence de recommandation."

---

## [01:00 – 02:30] LE PROBLÈME ET NOTRE APPROCHE

*[Ton : pédagogique, posé]*

"Le département marketing de The North Face avait identifié deux besoins concrets.

**Premier besoin :** Un système de recommandation. Sur chaque fiche produit, afficher 'Vous aimerez aussi ces produits similaires.' Simple en apparence, mais pour ça, il faut que la machine comprenne quels produits se ressemblent.

**Deuxième besoin :** Challenger la structure du catalogue. Est-ce que les catégories actuelles correspondent vraiment à la façon dont les clients pensent aux produits ? Existe-t-il des regroupements plus naturels qui amélioreraient la navigation ?

Notre approche : nous avons uniquement les descriptions textuelles d'environ 500 produits. Pas de notes de clients, pas d'historique d'achats — juste du texte.

Cela s'appelle de l'apprentissage **non supervisé** : l'algorithme apprend tout seul à trouver des patterns, sans qu'on lui dise d'avance ce qu'il doit chercher. C'est à la fois la force et la difficulté de ce type de projet."

---

## [02:30 – 04:00] L'EDA – CE QUE LES DONNÉES NOUS ONT APPRIS

*[Ton : analytique, montrez les insights]*

"Avant d'entraîner quoi que ce soit, on a d'abord exploré nos données.

Première observation : les descriptions sont longues et riches — en moyenne 150 mots par produit. C'est une bonne nouvelle pour le NLP : plus de texte signifie plus d'information pour distinguer les produits.

Deuxième observation — celle-ci est cruciale : quand on regarde les mots les plus fréquents dans le texte brut, on tombe sur 'and', 'the', 'a', 'for'... Des mots qui ne veulent rien dire sur le contenu d'un produit. Ce sont ce qu'on appelle des **mots vides**, ou stop words. Si on les laisse, ils vont polluer notre analyse.

Troisième observation, après un nettoyage rapide : les mots qui émergent vraiment sont 'polyester', 'nylon', 'recycled', 'water', 'jacket', 'pocket'... On voit déjà que les produits se distinguent par leur **matériau** et leur **type d'activité**.

Ça nous confirme que notre prétraitement va devoir être sérieux — et ça guide aussi le nombre de clusters à cibler."

---

## [04:00 – 05:30] LE PIPELINE DE PRÉTRAITEMENT

*[Ton : technique mais accessible]*

"Voyons maintenant comment on a préparé nos données pour les modèles.

Nos descriptions contiennent du HTML brut : des balises `<br>`, `<li>`, `<ul>`. La première étape, c'est de nettoyer tout ça. Ensuite, on n'a plus besoin que des mots alphabétiques.

Puis on passe par **spaCy**, une bibliothèque NLP professionnelle. Elle fait trois choses pour nous :
- Elle supprime les mots vides
- Elle **lemmatise** les mots : transformer 'wicking', 'wicks', 'wicked' en leur forme de base 'wick'. Ça réduit le bruit et regroupe les formes d'un même mot.
- Elle tokenise — elle découpe le texte en unités exploitables.

Enfin, on construit une matrice **TF-IDF**. Sans rentrer dans les détails mathématiques : TF-IDF pondère les mots. Il donne un poids élevé aux mots **rares et distinctifs** — comme 'H2No' ou 'Capilene', des technologies propres à The North Face — et un poids faible aux mots trop courants. On obtient une matrice de 500 lignes (produits) par 3000 colonnes (termes).

Tout ça est encapsulé dans un **Pipeline Scikit-Learn**, ce qui garantit la reproductibilité et l'absence de fuite de données."

---

## [05:30 – 07:00] PARTIE 1 : DBSCAN ET LA RECOMMANDATION

*[Ton : confiant, valorisez vos choix]*

"Avec cette matrice TF-IDF en main, on peut faire du clustering.

On a choisi **DBSCAN** — non pas K-Means, et c'est un choix délibéré. Pourquoi ? Parce que K-Means utilise la distance euclidienne, qui est inadaptée aux textes. Elle est sensible à la longueur des documents, pas à leur contenu. Et elle vous force à spécifier le nombre de clusters avant même d'analyser les données.

DBSCAN, lui, travaille avec la **similarité cosinus** — l'angle entre deux vecteurs de mots. Deux produits qui parlent tous les deux de 'waterproof' et 'nylon' auront un angle faible, donc une haute similarité, même si leurs descriptions sont de longueurs différentes.

Et DBSCAN détecte automatiquement les outliers — les produits qui ne ressemblent à rien d'autre dans le catalogue. C'est en soi une information précieuse.

Après avoir testé plusieurs configurations, on a trouvé un paramétrage qui donne une quinzaine de clusters cohérents. Les WordClouds le confirment visuellement : on voit clairement un cluster 'sous-vêtements techniques Capilene', un cluster 'pêche et wading', un cluster 'coton bio et casual'...

Sur cette base, la fonction de recommandation `find_similar_items` est simple et puissante : donnez-lui un ID produit, elle identifie son cluster, et retourne les 5 produits les plus similaires par similarité cosinus."

---

## [07:00 – 08:30] PARTIE 2 : LE TOPIC MODELING LSA

*[Ton : pédagogique, faites la distinction avec le clustering]*

"La troisième partie du projet est indépendante. Elle s'appelle le **Topic Modeling**, ou modélisation de sujets. Et ici, on utilise un algorithme qui s'appelle LSA — Latent Semantic Analysis — implémenté via TruncatedSVD dans Scikit-Learn.

Quelle est la différence avec le clustering ? Dans le clustering, chaque produit appartient à UN seul groupe. Dans le topic modeling, chaque produit peut appartenir à PLUSIEURS sujets simultanément.

Prenez une veste de randonnée recyclée respirante. Elle touche plusieurs sujets à la fois : les matériaux techniques, l'outdoor et l'aventure, la durabilité et l'éco-responsabilité. Le topic modeling capture cette richesse.

Avec 15 sujets, on explique environ 60 à 70% de la variance totale de notre corpus. Les WordClouds par sujet sont interprétables : chaque sujet a des mots caractéristiques qui lui donnent du sens.

Pour l'importance des variables — l'équivalent de la Feature Importance en supervisé — on regarde les **charges SVD** : les termes avec les valeurs les plus élevées dans chaque composante sont les plus représentatifs de ce sujet latent."

---

## [08:30 – 09:30] LES RÉSULTATS MÉTIER

*[Ton : orienté business, faites le lien avec la valeur]*

"Résumons ce que ces modèles apportent concrètement au business de The North Face.

Le clustering DBSCAN révèle une structure naturelle du catalogue que l'équipe marketing peut directement utiliser pour **réorganiser les filtres de navigation** sur le site web.

La fonction de recommandation est **prête à être déployée** via une API. Elle n'a pas besoin de données comportementales, elle fonctionne sur le contenu seul — ce qui élimine le problème du cold start.

Le topic modeling, lui, génère des insights pour le **SEO** : les mots-clés des topics latents peuvent alimenter les balises méta, les tags produits, et améliorer la découvrabilité sur Google.

Concernant les limites : environ 30% des produits sont classés comme outliers par DBSCAN. Ce ne sont pas des erreurs — ce sont des produits atypiques. Une revue humaine pourrait identifier les cas à recatégoriser."

---

## [09:30 – 10:00] CONCLUSION

*[Ton : enthousiaste, mémorable]*

"Pour conclure.

Ce projet montre comment le Machine Learning peut transformer un simple fichier CSV de descriptions textuelles en un système d'intelligence produit à valeur business réelle.

On a construit un pipeline NLP robuste avec spaCy et TF-IDF. On a appliqué DBSCAN avec la distance cosinus pour un clustering adapté au texte. On a développé un recommender system basé sur la similarité intra-cluster. Et on a extrait des sujets latents avec LSA pour challenger la structure du catalogue.

Le tout sans aucune variable cible, sans labels — en laissant les données parler d'elles-mêmes.

Je suis maintenant disponible pour vos questions. Merci beaucoup."

---

*[Durée estimée : 10 minutes – ajustez le débit selon le niveau de détail souhaité pour chaque section]*
