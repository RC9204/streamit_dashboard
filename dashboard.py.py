import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap

api_url = 'https://projet-17-rcuaz-heroku.herokuapp.com/predict'

df_modif_allege = pd.read_csv('df_modif_allege.csv')

train, test = train_test_split(df_modif_allege, test_size=0.2, random_state=0)
train.drop(columns=['index'], inplace=True)
test.drop(columns=['index'], inplace=True)
X_train = train.drop(columns=['TARGET'])
y_train = train['TARGET']
X_test = test.drop(columns=['TARGET'])
y_true = test['TARGET']

selected_client_id = st.selectbox('Selectionnez un client :', X_test['SK_ID_CURR'])
random_observation_df = X_test[X_test['SK_ID_CURR'] == selected_client_id]

data = random_observation_df.to_dict(orient='records')[0]

response = requests.post(api_url, json=data)
prediction = response.json()['prediction'][0]

client_id = random_observation_df['SK_ID_CURR'].values[0]

st.title(f"Dashboard de Credit")
st.markdown(f"### Num client: {client_id}")

credit_accepted = 'Non' if prediction == 1 else 'Oui'
st.markdown(f"### Credit accepte: {credit_accepted}")

st.markdown(f"### Score du client")
st.markdown(f"Score: {prediction:.2f}")