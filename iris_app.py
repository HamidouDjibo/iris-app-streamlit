import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Charger le modèle et les données
iris = load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

# Titre
st.title("Classification des Fleurs d'Iris")

# Description
st.write("""
         Cette application prédit la classe d'une fleur Iris en fonction des caractéristiques saisies.
         """)

# Entrées utilisateur
sepal_length = st.number_input('Longueur du sépale (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Largeur du sépale (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Longueur du pétale (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Largeur du pétale (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Préparation des données pour la prédiction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prédiction
if st.button("Prédire"):
    prediction = clf.predict(input_data)
    predicted_class = iris.target_names[prediction][0]
    st.write(f"La classe prédite est : {predicted_class}")
