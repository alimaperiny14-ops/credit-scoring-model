import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_data(file_path):
    """
    Charge le jeu de données à partir du chemin de fichier CSV spécifié.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'a pas été trouvé.")
    return pd.read_csv(file_path)

def create_preprocessing_pipeline(df):
    """
    Crée et renvoie un pipeline de prétraitement utilisant StandardScaler pour les caractéristiques numériques
    et OneHotEncoder pour les caractéristiques catégorielles.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée à analyser pour les types de caractéristiques.

    Returns:
        sklearn.pipeline.Pipeline: Le pipeline de prétraitement.
    """
    # Identifier les caractéristiques numériques et catégorielles
    # NOTE: Ces noms de colonnes sont des espaces réservés. Vous devrez les remplacer
    # par les noms de colonnes réels de votre jeu de données.
    # Pour la démonstration, supposons quelques colonnes basées sur des jeux de données de scoring de crédit typiques.
    
    # Exemples de caractéristiques numériques (à remplacer par les colonnes numériques réelles de votre jeu de données)
    numerical_features = ['Revenu', 'Age', 'Ratio_Endettement', 'Nombre_Credits', 'Montant_Credit']
    # Exemples de caractéristiques catégorielles (à remplacer par les colonnes catégorielles réelles de votre jeu de données)
    categorical_features = ['Type_Emploi', 'Secteur_Activite', 'Statut_Familial']

    # Filtrer les caractéristiques qui ne sont pas présentes dans le DataFrame
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    # Créer les étapes de prétraitement pour les caractéristiques numériques et catégorielles
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Créer un préprocesseur utilisant ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Conserver les autres colonnes non spécifiées (par exemple, cible, ID)
    )
    
    # Créer un pipeline avec juste le préprocesseur
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    return pipeline

if __name__ == '__main__':
    # Ce bloc est à des fins de test.
    # Dans un scénario réel, vous l'exécuteriez à partir de model_training.py ou Moda.ipynb
    
    # Créer un DataFrame factice pour les tests
    data = {
        'Revenu': [50000, 60000, 30000, 80000, 45000],
        'Age': [30, 45, 25, 50, 35],
        'Ratio_Endettement': [0.3, 0.5, 0.2, 0.6, 0.4],
        'Nombre_Credits': [2, 1, 3, 2, 1],
        'Montant_Credit': [10000, 20000, 5000, 30000, 15000],
        'Type_Emploi': ['CDI', 'CDD', 'CDI', 'Independant', 'CDI'],
        'Secteur_Activite': ['Finance', 'Tech', 'Retail', 'Finance', 'Tech'],
        'Statut_Familial': ['Marie', 'Celibataire', 'Marie', 'Celibataire', 'Marie'],
        'Target': [0, 1, 0, 1, 0] # Variable cible factice
    }
    dummy_df = pd.DataFrame(data)

    print("En-tête du DataFrame factice :")
    print(dummy_df.head())

    # Créer et ajuster le pipeline de prétraitement
    preprocessing_pipeline = create_preprocessing_pipeline(dummy_df)
    
    # Ajuster le pipeline sur les données factices (excluant la cible)
    X_dummy = dummy_df.drop('Target', axis=1)
    preprocessing_pipeline.fit(X_dummy)

    print("\nPipeline de prétraitement créé avec succès.")
    print(preprocessing_pipeline)

    # Vous pouvez transformer les données pour voir la sortie
    X_transformed = preprocessing_pipeline.transform(X_dummy)
    print("\nForme des données transformées :", X_transformed.shape)
    # Remarque : X_transformed sera un tableau numpy, pas un DataFrame, après OneHotEncoder.
