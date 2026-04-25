import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Importer les fonctions de data_preprocessing.py
from .data_preprocessing import load_data, create_preprocessing_pipeline

def train_and_save_model(data_path, model_output_path):
    """
    Charge les données, crée un pipeline de prétraitement et de modèle, entraîne le modèle,
    et sauvegarde le pipeline complet.

    Args:
        data_path (str): Chemin vers le fichier CSV du jeu de données brut.
        model_output_path (str): Chemin où le pipeline de modèle entraîné sera sauvegardé.
    """
    print(f"Chargement des données depuis {data_path}...")
    df = load_data(data_path)

    # Supposons que 'Target' est le nom de votre variable cible
    # Vous devrez peut-être ajuster cela en fonction de votre jeu de données réel
    if 'Target' not in df.columns:
        raise ValueError("La colonne cible 'Target' n'a pas été trouvée dans le jeu de données. Veuillez vérifier vos données.")
    
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Création du pipeline de prétraitement...")
    preprocessing_pipeline = create_preprocessing_pipeline(X_train)

    # Créer le pipeline complet : prétraitement + modèle
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))
    ])

    print("Entraînement du modèle...")
    model_pipeline.fit(X_train, y_train)

    print("Évaluation du modèle sur l'ensemble de test...")
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Score AUC du modèle : {auc_score:.4f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    print(f"Sauvegarde du modèle entraîné dans {model_output_path}...")
    joblib.dump(model_pipeline, model_output_path)
    print("Modèle sauvegardé avec succès.")

if __name__ == '__main__':
    # Définir les chemins
    DATA_FILE_PATH = 'data/raw/dataset_scoring_credit_900k.csv'
    MODEL_SAVE_PATH = 'models/best_credit_scoring_model.pkl'

    # Toujours créer un jeu de données factice à des fins de test
    print(f"Création/Écrasement d'un jeu de données factice à {DATA_FILE_PATH} pour les tests...")
    dummy_data = {
        'Revenu': [50000, 60000, 30000, 80000, 45000, 70000, 35000, 90000, 55000, 40000],
        'Age': [30, 45, 25, 50, 35, 40, 28, 55, 32, 48],
        'Ratio_Endettement': [0.3, 0.5, 0.2, 0.6, 0.4, 0.45, 0.25, 0.7, 0.35, 0.55],
        'Nombre_Credits': [2, 1, 3, 2, 1, 2, 3, 1, 2, 1],
        'Montant_Credit': [10000, 20000, 5000, 30000, 15000, 25000, 7000, 40000, 18000, 22000],
        'Type_Emploi': ['CDI', 'CDD', 'CDI', 'Independant', 'CDI', 'CDI', 'CDD', 'Independant', 'CDI', 'CDD'],
        'Secteur_Activite': ['Finance', 'Tech', 'Retail', 'Finance', 'Tech', 'Retail', 'Finance', 'Tech', 'Retail', 'Finance'],
        'Statut_Familial': ['Marie', 'Celibataire', 'Marie', 'Celibataire', 'Marie', 'Celibataire', 'Marie', 'Celibataire', 'Marie', 'Celibataire'],
        'Target': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1] # Variable cible factice
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(DATA_FILE_PATH, index=False)
    print("Jeu de données factice créé/écrasé.")

    train_and_save_model(DATA_FILE_PATH, MODEL_SAVE_PATH)
