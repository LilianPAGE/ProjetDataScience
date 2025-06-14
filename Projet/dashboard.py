import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib  # pour charger un modèle

# --- Chargement des données ---
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# --- Chargement du modèle de prédiction ---
model = joblib.load("modele_prix.joblib")  # Assurez-vous que ce fichier existe

# --- Sidebar - Saisie utilisateur ---
st.sidebar.header("🎯 Saisissez les caractéristiques du bien")

surface = st.sidebar.slider("Surface habitable au-dessus du sol (pieds²)", 300, 5000, 1000)
nombre_pieces = st.sidebar.slider("Nombre total de pièces (hors sous-sol et salles de bain)", 1, 15, 6)
etage = st.sidebar.slider("Nombre d'étages", 1, 3, 1)  # pas dans les variables, à ajuster si besoin
localisation = st.sidebar.selectbox("Quartier", df["Neighborhood"].unique())

# --- Données pour prédiction ---
# Adapter les colonnes en fonction du modèle et dataset
# Ici je suppose que ton modèle attend des colonnes comme 'GrLivArea' pour surface,
# 'TotRmsAbvGrd' pour nombre de pièces, et 'Neighborhood' pour localisation

input_data = pd.DataFrame({
    "GrLivArea": [surface],
    "TotRmsAbvGrd": [nombre_pieces],
    "Neighborhood": [localisation]
})

# --- Encodage si nécessaire (ex: one-hot encoding) ---
input_encoded = pd.get_dummies(input_data).reindex(columns=model.feature_names_in_, fill_value=0)

# --- Prédiction ---
prix_estime = model.predict(input_encoded)[0]

st.title("🏠 Dashboard Immobilier Interactif")
st.subheader("💡 Prédiction de prix")
st.metric("Prix estimé ($)", f"{prix_estime:,.0f} $")

# --- Graphique 1 : Impact des variables ---
st.subheader("📊 Analyse de l’impact des variables")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.box(df, x="TotRmsAbvGrd", y="SalePrice", title="Prix par nombre de pièces")
    st.plotly_chart(fig1)

with col2:
    fig2 = px.scatter(df, x="GrLivArea", y="SalePrice", color="Neighborhood", title="Prix en fonction de la surface habitable")
    st.plotly_chart(fig2)


# --- Footer ---
st.markdown("---")
st.markdown("📍 Projet de visualisation et prédiction immobilière avec Streamlit")
