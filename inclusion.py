import streamlit as st
import pandas as pd
#import pickle
#import sklearn
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger le modèle
#model_path = 'models/votre_modele.pkl'
clf = joblib.load('modeleclf.joblib')

# Charger le modèle préalablement entraîné
#with open("modeleclf.pkl", 'rb') as model_file:
#    clf = pickle.load(model_file)

# Charger vos données (vous pouvez adapter cela si vous souhaitez charger vos données depuis un fichier CSV)
data = pd.read_csv('Financial_inclusion_dataset.csv')

# Encoder les caractéristiques catégorielles avec LabelEncoder
#label_encoder = joblib.load('models/label_encoder.pkl')
#for col in data.select_dtypes(include=['object']).columns:
#    data[col] = label_encoder.fit_transform(data[col])

# fonction de prediction
def predict_bank_account(feature):
    pred = clf.predict(feature)
    return pred[0]

def main():
    # Page d'accueil de l'application
    st.title("Application de Prédiction d'Inclusion Financière")

    # Formulaire pour saisir les caractéristiques
    st.header("Saisissez les caractéristiques :")

    country_options = data['country'].unique()
    country = st.selectbox("Pays", country_options)

    year = st.number_input("Année", min_value=int(data['year'].min()), max_value=int(data['year'].max()))

    location_type_options = data['location_type'].unique()
    location_type = st.selectbox("Type de localisation", location_type_options)

    cellphone_access_options = data['cellphone_access'].unique()
    cellphone_access = st.selectbox("Accès au téléphone portable", cellphone_access_options)

    household_size = st.number_input("Taille du ménage", min_value=int(data['household_size'].min()), max_value=int(data['household_size'].max()))

    age_of_respondent = st.number_input("Âge du répondant", min_value=int(data['age_of_respondent'].min()), max_value=int(data['age_of_respondent'].max()))

    gender_of_respondent_options = data['gender_of_respondent'].unique()
    gender_of_respondent = st.selectbox("Genre du répondant", gender_of_respondent_options)

    relationship_with_head_options = data['relationship_with_head'].unique()
    relationship_with_head = st.selectbox("Relation avec le chef de ménage", relationship_with_head_options)

    marital_status_options = data['marital_status'].unique()
    marital_status = st.selectbox("Statut matrimonial", marital_status_options)

    education_level_options = data['education_level'].unique()
    education_level = st.selectbox("Niveau d'éducation", education_level_options)

    job_type_options = data['job_type'].unique()
    job_type = st.selectbox("Type d'emploi", job_type_options)

    # Ajouter un bouton de soumission
    submit_button = st.button("Soumettre")

    # Vérifier si le bouton a été pressé
    if submit_button:
        # Préparer les données pour la prédiction
        features = [country, year, location_type, cellphone_access, household_size, age_of_respondent,
                    gender_of_respondent, relationship_with_head, marital_status, education_level, job_type]
        label_encoder = LabelEncoder()
        # Encoder les valeurs avec LabelEncoder
        features_encoded = [label_encoder.fit_transform([feature])[0] if isinstance(feature, str) else feature for feature in features]

        # Préparer les données pour la prédiction
        input_data = [features_encoded]

        # Faire la prédiction
        prediction = predict_bank_account(input_data)

        # Afficher la prédiction
        st.header("Résultat de la prédiction :")
        st.write(f"La personne est prédite {'avoir' if prediction else 'ne pas avoir'} un compte bancaire.")

if __name__ == '__main__':
    main()
