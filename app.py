import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Charger le modèle entraîné
MODEL_PATH = 'models/best_credit_scoring_model.joblib'

@st.cache_resource
def load_model(path):
    """Charge le pipeline de modèle entraîné."""
    if not os.path.exists(path):
        st.error(f"Fichier modèle non trouvé à {path}. Veuillez vous assurer que le modèle est entraîné et sauvegardé.")
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

st.set_page_config(page_title="Application de Scoring Crédit", layout="centered")

st.title("Application de Scoring Crédit")
st.write("Cette application prédit le risque de défaut de crédit d'un client.")

if model is None:
    st.stop()

# Définir les caractéristiques d'entrée basées sur les caractéristiques attendues du modèle
# Celles-ci doivent correspondre aux caractéristiques utilisées lors de l'entraînement et du prétraitement du modèle
numerical_features = ['Revenu', 'Age', 'Ratio_Endettement', 'Nombre_Credits', 'Montant_Credit']
categorical_features = ['Type_Emploi', 'Secteur_Activite', 'Statut_Familial']

# Exemples de catégories pour la boîte de sélection (à remplacer par les catégories réelles de votre jeu de données)
type_emploi_options = ['CDI', 'CDD', 'Interim', 'Independant', 'Retraite', 'Etudiant']
secteur_activite_options = ['Finance', 'Tech', 'Retail', 'Sante', 'Education', 'Autre']
statut_familial_options = ['Marie', 'Celibataire', 'Divorce', 'Veuf']

st.header("Informations Client")

# Champs de saisie pour les caractéristiques numériques
input_data = {}
for feature in numerical_features:
    if feature in ['Age', 'Nombre_Credits']:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ')}", value=0, step=1, format="%d")
    else:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ')}", value=0.0, format="%.2f")

# Champs de saisie pour les caractéristiques catégorielles
input_data['Type_Emploi'] = st.selectbox("Type d'emploi", options=type_emploi_options)
input_data['Secteur_Activite'] = st.selectbox("Secteur d'activité", options=secteur_activite_options)
input_data['Statut_Familial'] = st.selectbox("Statut familial", options=statut_familial_options)

# Convertir les données d'entrée en DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Prédire le risque"):
    try:
        # Effectuer la prédiction
        prediction_proba = model.predict_proba(input_df)[:, 1][0] # Probabilité de défaut
        
        # Convertir la probabilité en pourcentage
        default_probability_percent = prediction_proba * 100

        # Logique de décision
        # Vous voudrez peut-être définir un seuil pour la probabilité de défaut
        # Par exemple, si prob > 0.5, alors 'Refusé'
        threshold = 0.5 
        decision = "Crédit Refusé" if prediction_proba >= threshold else "Crédit Accordé"
        decision_color = "red" if prediction_proba >= threshold else "green"

        # Calculer le score sur 1000 points (inverse de la probabilité de défaut)
        # Un score plus élevé signifie un risque plus faible
        credit_score = int((1 - prediction_proba) * 1000)

        st.subheader("Résultats de la Prédiction")
        st.markdown(f"Probabilité de défaut: **{default_probability_percent:.2f}%**")
        st.markdown(f"Décision: <span style='color:{decision_color}; font-weight:bold;'>{decision}</span>", unsafe_allow_html=True)
        st.markdown(f"Score de risque: **{credit_score} / 1000**")

        # Jauge visuelle pour le score
        st.progress(credit_score / 1000)

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction: {e}")
        st.warning("Veuillez vérifier que toutes les informations sont correctement saisies et que le modèle est bien chargé.")
