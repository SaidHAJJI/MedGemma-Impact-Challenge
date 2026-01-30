# MedGemma Impact Challenge - Android Triage App

## Description
Une application Android "Privacy-First" pour le triage médical d'urgence, fonctionnant 100% hors ligne grâce à MedGemma (via MediaPipe).

## Structure du Projet
- `android_app/` : Le code source de l'application Android (Kotlin + Jetpack Compose).
- `venv/` : Environnement Python pour les scripts auxiliaires.

## Installation & Test (Android)

### Pré-requis
1.  **Android Studio** (Koala ou plus récent recommandé).
2.  Un appareil Android physique (Mode Développeur activé) - L'émulateur peut être lent pour le LLM.

### Étape 1 : Obtenir le Modèle
Vous devez télécharger une version de Gemma compatible MediaPipe (format `.bin` ou `.task`).
- Téléchargez `gemma-2b-it-gpu-int4.bin` depuis Kaggle ou Hugging Face (Google AI Edge).
- Renommez ce fichier en `medgemma.bin`.

### Étape 2 : Placer le Modèle sur l'appareil
L'application cherche le fichier dans le stockage interne de l'application.
Une fois l'application installée (via Android Studio), utilisez l'outil **Device File Explorer** dans Android Studio :
1.  Allez dans `View > Tool Windows > Device File Explorer`.
2.  Naviguez vers `/data/data/com.example.medgemma/files/`.
3.  Faites un clic droit > **Upload...** et sélectionnez votre fichier `medgemma.bin`.

### Étape 3 : Lancer l'App
Ouvrez le dossier `android_app` dans Android Studio et lancez le build.

## Roadmap
- [x] Squelette de l'application (Texte).
- [x] Intégration Vocale (Speech-to-Text).
- [x] Optimisation du Prompt pour le contexte médical (V3 SafetyFirst).
- [x] Triage en 2 étapes (Questions -> Rapport).
- [ ] Export du rapport en PDF (Android).