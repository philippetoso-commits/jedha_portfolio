# FAQ : Conversion Rate Challenge - Soutenance & Technique
*Ce document anticipe les questions techniques et métier que le jury pourrait vous poser suite à votre présentation.*

---

### Questions sur la Méthodologie
**Q : Pourquoi avoir choisi XGBoost plutôt qu'un Random Forest ou une Régression Logistique ?**
*R :* XGBoost est un algorithme de Gradient Boosting qui construit des arbres de manière séquentielle pour corriger les erreurs des précédents. Il gère mieux les relations non-linéaires que la Régression Logistique et offre plus de finesse de réglage que le Random Forest. Surtout, c'est l'un des rares à permettre l'implémentation native de contraintes de monotonie.

**Q : C'est quoi exactement une "Contrainte de Monotonie" ?**
*R :* C'est une règle mathématique qu'on impose au modèle. Par exemple, pour la variable `Total_Pages_Visited`, on impose une relation monotone croissante : le modèle a l'interdiction de prédire une baisse de conversion si le nombre de pages augmente. Cela évite l'overfitting sur des bruits locaux du dataset.

**Q : Pourquoi avoir choisi un seuil de 0.385 au lieu du standard 0.5 ?**
*R :* Le seuil de 0.5 est optimal pour l'Accuracy (justesse globale). Mais ici, nous étions jugés sur le F1-Score (équilibre entre Précision et Rappel). Dans un dataset déséquilibré (peu de conversions), abaisser légèrement le seuil permet de capturer plus de convertis (meilleur Rappel) sans trop dégrader la Précision.

---

### Questions sur les Données (EDA)
**Q : Quelles étaient les variables les plus importantes ?**
*R :* Sans aucun doute `Total_Pages_Visited`. C'est le signal "fort". Ensuite viennent l'âge (relation inverse) et le fait d'être un "New User" ou non. Les sources de trafic (Direct, SEO, Ads) avaient un impact beaucoup plus marginal.

**Q : Avez-vous traité des valeurs manquantes ou des outliers ?**
*R :* Le dataset était relativement propre. Cependant, j'ai vérifié la cohérence de l'âge (pas d'utilisateurs de 150 ans) et j'ai traité les variables catégorielles (Pays, Source) via un OneHotEncoder pour les rendre lisibles par XGBoost.

---

### Questions sur la Performance
**Q : Vous avez gagné avec un modèle simple. Est-ce que le Stacking n'aurait pas été meilleur ?**
*R :* J'ai testé des architectures de Stacking (mélange de CatBoost, LightGBM et XGBoost). Bien qu'elles soient performantes sur le train set, elles perdaient en généralisation sur le test set (overfitting). La "Stratégie ADN" (un seul modèle contraint) s'est avérée plus robuste et plus simple à maintenir en production.

**Q : Si vous aviez plus de temps, comment amélioreriez-vous ce score ?**
*R :* J'explorerais davantage le *Feature Engineering*, par exemple en créant des interactions entre le pays et la source de trafic. On pourrait aussi envisager un modèle spécifique par segment de pays si les comportements d'achat diffèrent trop d'une zone géographique à l'autre.

---

### Questions Métier (Business)
**Q : Quelle recommandation concrète donneriez-vous à l'équipe Marketing ?**
*R :* 1. Focus sur l'engagement : Tout levier augmentant le nombre de pages vues (recommandations produits, UX fluide) augmentera mécaniquement la conversion. 
2. Retargeting : Cibler prioritairement les utilisateurs qui ont visité plus de 10 pages mais n'ont pas encore acheté, car leur probabilité de conversion est statistiquement immense.
