# Projet de Scoring Crédit avec Streamlit

Ce projet vise à déployer un modèle de scoring crédit sous la forme d'une application web interactive en utilisant Streamlit. L'application permet à un conseiller bancaire de saisir les informations d'un client et d'obtenir une prédiction de risque de défaut, ainsi qu'un score de risque sur 1000 points.

## Structure du Projet

Le projet est organisé comme suit :

```
.
├── Moda.ipynb                        # Notebook pour l'EDA et l'orchestration 
├── requirements.txt                  # Liste des dépendances Python
├── README.md                         # Documentation du projet
├── data/                             # Contient les données brutes
│   └── raw/
│       └── dataset_scoring_credit_900k.csv # Fichier de données (créer moi meme)
├── src/                              # Code source des modules Python
│   ├── __init__.py                   # Initialise le dossier comme un package Python
│   ├── data_preprocessing.py         # Fonctions pour le chargement et le prétraitement des données
│   └── model_training.py             # Fonctions pour l'entraînement, l'évaluation et la sauvegarde des modèles
├── models/                           # Dossier pour sauvegarder les modèles entraînés
│   └── best_credit_scoring_model.joblib # Modèle de scoring crédit entraîné
└── app.py                            # Application web Streamlit
```

## Étapes de Mise en Œuvre

### 1. Préparation du Modèle (Backend)

Le script `src/model_training.py` est responsable de l'entraînement du modèle de régression logistique. Il utilise les fonctions de `src/data_preprocessing.py` pour charger et prétraiter les données. Le modèle entraîné est ensuite sauvegardé sous `models/best_credit_scoring_model.joblib`.

Pour entraîner le modèle, assurez-vous d'avoir un fichier `dataset_scoring_credit_900k.csv` dans `data/raw/`. Si ce fichier n'existe pas, le script `model_training.py` créera un jeu de données factice pour les tests.

Exécuter l'entraînement :

```bash
python -m src.model_training
```

### 2. Création de l'Application Streamlit (Frontend)

Le fichier `app.py` contient le code de l'application web Streamlit. Il charge le modèle sauvegardé et fournit une interface utilisateur pour saisir les informations client et obtenir des prédictions.

Pour lancer l'application Streamlit localement :

```bash
streamlit run app.py
```

L'application sera accessible via votre navigateur web, généralement à `http://localhost:8501`.

### Comment utiliser l'application

Une fois l'application Streamlit lancée (via `streamlit run app.py`), suivez ces étapes :

1.  **Saisir les informations client** : Dans la section "Informations Client", vous trouverez des champs de saisie pour les variables numériques (Revenu, Âge, Ratio d'endettement, Nombre de crédits, Montant du crédit) et des listes déroulantes pour les variables catégorielles (Type d'emploi, Secteur d'activité, Statut familial). Remplissez ces champs avec les données du client.
2.  **Lancer la prédiction** : Cliquez sur le bouton "Prédire le risque".
3.  **Consulter les résultats** : L'application affichera alors :
    *   La probabilité de défaut (en pourcentage).
    *   Une décision : "Crédit Accordé" (en vert) ou "Crédit Refusé" (en rouge).
    *   Un score de risque sur 1000 points, accompagné d'une jauge visuelle. Un score plus élevé indique un risque plus faible.

### 3. Déploiement sur Streamlit Cloud

Pour déployer l'application sur Streamlit Cloud, suivez ces étapes :

1.  **Dépendances** : Assurez-vous que toutes les bibliothèques nécessaires sont listées dans `requirements.txt`.
    ```
    streamlit
    pandas
    scikit-learn
    joblib
    ```
2.  **GitHub** : Déposez tous les fichiers du projet (`app.py`, `requirements.txt`, `src/`, `models/`, `data/raw/dataset_scoring_credit_900k.csv` et `Moda.ipynb`) dans un dépôt GitHub.
3.  **Streamlit Cloud** : Connectez-vous à Streamlit Cloud (share.streamlit.io) et liez votre compte à votre dépôt GitHub. Déployez l'application en sélectionnant le dépôt et le fichier `app.py` comme point d'entrée.

## Variables Utilisées

Les variables d'entrée attendues par le modèle et l'application Streamlit sont les suivantes (ces noms sont basés sur les exemples fournis et peuvent nécessiter des ajustements en fonction de votre jeu de données réel) :

**Numériques :**

- Revenu
- Age
- Ratio_Endettement
- Nombre_Credits
- Montant_Credit

**Catégorielles :**

- Type_Emploi (ex: CDI, CDD, Interim, Independant, Retraite, Etudiant)
- Secteur_Activite (ex: Finance, Tech, Retail, Sante, Education, Autre)
- Statut_Familial (ex: Marie, Celibataire, Divorce, Veuf)

## Auteur ALIMA PERINY

Ce projet a été réalisé dans le cadre du MBA1 Finance Digitale à l'ISM Dakar, Année Académique 2025-2026.

---
