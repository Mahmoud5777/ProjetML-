from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import sys

# 1. Permettre à Flask de trouver ton fichier preprocessing.py dans le dossier src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '../src')
sys.path.insert(0, src_path)

from preprocessing import clean_and_preprocess

app = Flask(__name__)

# 2. Charger le modèle et les colonnes sauvegardés lors de l'entraînement
MODEL_PATH = os.path.join(current_dir, '../models/production_model.joblib')
COLUMNS_PATH = os.path.join(current_dir, '../models/model_columns.joblib')

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

@app.route('/')
def home():
    # Affiche la page web d'accueil au démarrage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données tapées par l'utilisateur dans le formulaire
        form_data = request.form.to_dict()
        df_input = pd.DataFrame([form_data])
        
        # Convertir le texte du web en nombres pour l'algorithme
        for col in df_input.columns:
            try:
                df_input[col] = pd.to_numeric(df_input[col])
            except ValueError:
                pass
                
        # Appliquer ton code de nettoyage (preprocessing.py)
        df_processed = clean_and_preprocess(df_input)
        
        # Aligner les colonnes exactement comme le modèle l'exige
        df_final = df_processed.reindex(columns=model_columns, fill_value=0)
        
        # Faire la prédiction
        prediction = model.predict(df_final)[0]
        
        # Préparer la réponse visuelle
        if prediction == 1:
            result = "⚠️ Attention : Risque de Churn (Départ imminent)"
            color = "#d9534f" # Rouge
        else:
            result = "✅ Client Fidèle (Pas de risque)"
            color = "#5cb85c" # Vert
            
        return render_template('index.html', prediction_text=result, text_color=color)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Erreur : {str(e)}", text_color="red")

if __name__ == '__main__':
    app.run(debug=True)