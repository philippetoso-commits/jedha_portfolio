# Guide de Présentation : ALPR Engine (2 min)

Ce guide optimise votre discours pour un débit naturel tout en couvrant les points essentiels. **Chaque section dure environ 20 secondes.**

## Introduction (0:00 - 0:15)
"Bonjour à tous. Je vais vous présenter notre application ALPR, un système complet de gestion d'accès par reconnaissance de plaques, conçu pour la production. "

## 1. Pipeline Technique (0:15 - 0:40)
* **Action :** Allez sur l'onglet **Image Processing**. Téléchargez une image.
* **Discours :** "Le cœur du système est notre pipeline d'IA. Contrairement à une boîte noire, nous visualisons ici chaque étape : l'image brute, la détection précise par YOLOv8, l'extraction de la zone d'intérêt, et enfin l'OCR. Nous avons même intégré une 'Error Gallery' pour documenter les cas limites, assurant une transparence totale sur les performances du modèle. "

## 2. Apprentissage Actif & Vidéo (0:40 - 1:05)
* **Action :** Passez rapidement par **Settings** puis **Video Processing**.
* **Discours :** "Pour l'amélioration continue, l'onglet Settings permet d'exporter les erreurs vers Hugging Face pour l'apprentissage actif. C'est une boucle rétroactive directe. Le système gère aussi la vidéo avec trois modes, allant de l'échantillonnage rapide à l'annotation complète pour une traçabilité visuelle de chaque accès. "

## 3. Administration & Business (1:05 - 1:30)
* **Action :** Allez sur l'onglet **Administration Clients**.
* **Discours :** "Côté métier, nous gérons ici nos 80 résidents. Le tableau de bord affiche les indicateurs clés : accès actifs, abonnements et blocages. Le moteur de recherche permet une gestion granulaire : on peut ajouter, modifier ou supprimer un client et ses droits d'accès en quelques clics. "

## 4. Historique & Qualité (1:30 - 1:50)
* **Action :** Montrez l'onglet **History** puis **Incohérence Système/Terrain**.
* **Discours :** "La traçabilité est assurée par un journal chronologique complet (History). Mais nous allons plus loin avec le 'Contrôle Qualité' : l'application détecte si la marque du véhicule vu par la caméra correspond à celle en base de données. C'est une sécurité supplémentaire contre les usurpations ou les changements de véhicules non déclarés. "

## Conclusion (1:50 - 2:00)
"En résumé, c'est une solution prête pour le terrain, alliant IA de pointe, gestion de données robuste et contrôle qualité proactif. Merci pour votre attention. "

---
### Conseils de pro
- **Temps :** ~270 mots. À lire à un rythme posé.
- **Micro-interactions :** Survolez les éléments quand vous les mentionnez pour rendre la démo vivante.
- **Confiance :** Si une détection prend du temps, commentez : "Ici, l'IA analyse chaque pixel pour garantir la précision. "
