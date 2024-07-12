import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap

api_url = "https://projet-17-rcuaz-heroku.herokuapp.com/predict"

df_modif_allege = pd.read_csv('df_modif_allege.csv')
selected_client_id = st.selectbox('Selectionnez un client :', df_modif_allege['SK_ID_CURR'])

if st.button('Obtenir les prédictions'):
    client_data = df_modif_allege[df_modif_allege['SK_ID_CURR'] == selected_client_id]
    data = client_data.to_dict(orient='records')[0]

    response = requests.post(api_url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prédiction : {result['prediction']}")
        st.write(f"Probabilité : {result['probability']}")
    else:
        st.write("Erreur lors de l'obtention des prédictions.")