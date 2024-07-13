import requests
import json
import streamlit as st
import pandas as pd
import shap
import random
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb

url = 'https://projet-17-rcuaz-heroku-0e22e8a46c1f.herokuapp.com/predict'
headers = {'Content-Type': 'application/json'}

modeleP17 = pickle.load(open('modele_P17.sav', 'rb'))
df_modif_allege = pd.read_csv('df_modif_allege.csv')

sub_X_test = df_modif_allege.drop(columns=['TARGET','index'])

explainer_glob = shap.TreeExplainer(modeleP17)
shap_values_glob = explainer_glob.shap_values(sub_X_test)

global_importance_scores = abs(shap_values_glob).mean(axis=0)
importance_df = pd.DataFrame({'Feature': sub_X_test.columns, 'Importance': global_importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_20_features = importance_df.head(20)['Feature'].tolist()

selected_client_id = st.selectbox('Selectionnez un client :', sub_X_test['SK_ID_CURR'])
random_observation_df = sub_X_test[sub_X_test['SK_ID_CURR'] == selected_client_id]

data = {'SK_ID_CURR': selected_client_id}
response = requests.post(url, data=json.dumps(data), headers=headers)

y_pred_proba = response.json()
threshold = 0.4
credit_accepted = 'Non' if y_pred_proba >= threshold else 'Oui'

explainer = shap.TreeExplainer(modeleP17)
shap_values = explainer.shap_values(random_observation_df)

client_id = random_observation_df['SK_ID_CURR'].values[0]

st.title(f"Dashboard de Credit")
st.markdown(f"### Num client: {client_id}")

st.markdown(f"### Credit accepte: {credit_accepted}")

st.markdown(f"### Score du client")
st.markdown(f"Score: {y_pred_proba:.2f}")
threshold = 0.4
fig, ax = plt.subplots(figsize=(6, 1))
ax.barh([0], [y_pred_proba], color=['green' if credit_accepted == 'Oui' else 'red'])
ax.axvline(x=threshold, color='blue', linestyle='--')
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xticks([0, threshold, 1])
ax.set_xticklabels(['0', '0.4', '1'])
st.pyplot(fig)

st.markdown("### Feature Importance Globale")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values_glob, sub_X_test, plot_type="dot", show=False)
st.pyplot(fig_summary)

def st_shap(plot, height=None):
    """Display a SHAP plot in Streamlit."""
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.markdown("### Feature Importance Locale")
fig_force = shap.force_plot(explainer.expected_value, shap_values[0, :], random_observation_df.iloc[0, :])
st_shap(fig_force, height=150)

local_shap_values = shap_values[0, :]
local_importance_df = pd.DataFrame({'Feature': random_observation_df.columns, 'SHAP Value': local_shap_values})
local_importance_df = local_importance_df.sort_values(by='SHAP Value', ascending=False)
top_5_local_importance = local_importance_df.head(5)

st.table(top_5_local_importance)

selected_feature = st.selectbox('Selectionnez une 1ere feature :', top_20_features)
selected_feature2 = st.selectbox('Selectionnez une 2eme feature :', top_20_features)

if selected_feature:
    feature_data = sub_X_test[selected_feature]
    fig_client_value, ax_client_value = plt.subplots()
    feature_values = feature_data.values
    client_value = random_observation_df[selected_feature].values[0]
    ax_client_value.hist(feature_values, alpha=0.5, label='Distribution')
    ax_client_value.axvline(x=client_value, color='r', linestyle='--', label='Position du client')
    plt.xlabel(selected_feature)
    plt.ylabel('Count')
    plt.title(f'Positionnement de la valeur du client pour {selected_feature}')
    plt.legend()
    st.pyplot(fig_client_value)

if selected_feature2:
    feature_data2 = sub_X_test[selected_feature2]
    fig_client_value2, ax_client_value2 = plt.subplots()
    feature_values2 = feature_data2.values
    client_value2 = random_observation_df[selected_feature2].values[0]
    ax_client_value2.hist(feature_values2, alpha=0.5, label='Distribution')
    ax_client_value2.axvline(x=client_value2, color='r', linestyle='--', label='Position du client')
    plt.xlabel(selected_feature2)
    plt.ylabel('Count')
    plt.title(f'Positionnement de la valeur du client pour {selected_feature2}')
    plt.legend()
    st.pyplot(fig_client_value2)

if selected_feature and selected_feature2:
    accepted_clients = sub_X_test[modeleP17.predict(sub_X_test) < threshold]
    rejected_clients = sub_X_test[modeleP17.predict(sub_X_test) >= threshold]
    
    fig_bivariate, ax_bivariate = plt.subplots()
    ax_bivariate.scatter(accepted_clients[selected_feature], accepted_clients[selected_feature2], color='green', alpha=0.5, label='Credit accepte')
    ax_bivariate.scatter(rejected_clients[selected_feature], rejected_clients[selected_feature2], color='red', alpha=0.5, label='Credit non accepte')
    ax_bivariate.scatter(random_observation_df[selected_feature], random_observation_df[selected_feature2], color='blue', marker='*', s=200, label='Client selectionne')
    plt.xlabel(selected_feature)
    plt.ylabel(selected_feature2)
    plt.title(f'Analyse bi-variee entre {selected_feature} et {selected_feature2}')
    plt.legend()
    st.pyplot(fig_bivariate)