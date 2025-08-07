import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Configuration de la page
st.set_page_config(page_title="Prédiction Crédit Client", layout="wide")

# CSS pour fond vert
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f9f9dc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Chargement du modèle
model = joblib.load("modele_rf1.joblib")

# En-tête avec logo et auteur
col_logo, col_titre = st.columns([1, 4])

with col_logo:
    if os.path.exists("logo_3dc.png"):
        st.image("logo_3dc.png", width=350)
    else:
        st.warning("Logo non trouvé.")
    st.markdown("**Développé par Jimy Ebanga**")

with col_titre:
    st.markdown("<h1 style='color:#1a73e8;'>📊 Tableau de bord de prédiction de crédit</h1>", unsafe_allow_html=True)

# Sidebar : Entrée utilisateur
with st.sidebar:
    st.markdown("## 🧾 Paramètres du client")
    age = st.slider("Âge", 18, 70, 35)
    revenu = st.number_input("Revenu mensuel (FCFA)", min_value=50000, max_value=2000000, value=300000, step=50000)
    charges = st.number_input("Charges mensuelles (FCFA)", min_value=10000, max_value=1500000, value=100000, step=50000)
    duree = st.slider("Durée du crédit (mois)", 6, 36, 12)
    montant = st.number_input("Montant demandé (FCFA)", min_value=50000, max_value=20000000, value=2000000, step=100000)
    garantie = st.selectbox("Type de garantie", ["aucune", "personnelle", "réelle"])
    garantie_enc = {"aucune": 0, "personnelle": 1, "réelle": 2}[garantie]
    cashflow = round(0.7 * (revenu - charges), 2)
    st.markdown("---")
    st.write(f"💡 Cashflow (70%) = **{cashflow:,.0f} FCFA**")

# Corps principal
col1, col2 = st.columns([2, 2])

with col1:
    st.markdown("### 🎯 Données client")
    st.metric("Âge", f"{age} ans")
    st.metric("Revenu net", f"{revenu - charges:,.0f} FCFA")
    st.metric("Durée", f"{duree} mois")
    st.metric("Montant demandé", f"{montant:,.0f} FCFA")
    st.metric("Garantie", garantie)

    # Graphique cashflow vs montant
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Cashflow sur durée", "Montant demandé"], [cashflow * duree, montant], color=["green", "red"])
    ax.set_ylabel("Montant total (FCFA)")
    ax.set_title("💰 Comparaison cashflow vs montant demandé")
    st.pyplot(fig)

with col2:
    st.markdown("### 🤖 Résultat de la prédiction")

    if st.button("Lancer la prédiction"):
        features = np.array([[age, revenu, charges, cashflow, montant, duree, garantie_enc]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if prediction == 0:
            st.success("✅ Crédit validé")
        else:
            st.error("❌ Crédit non validé")

        st.markdown(f"**📈 Probabilité de refus : `{proba*100:.2f}%`**")
    else:
        st.info("Cliquez sur le bouton pour lancer la prédiction.")

