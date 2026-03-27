import pandas as pd
import os
from preprocessing import clean_and_preprocess
from utils import load_model, load_columns

def predict_batch(input_file, output_file):
    print("=== PRÉDICTION EN LOT (BATCH) ===")
    
    # 1. Charger les nouvelles données
    print(f"1. Chargement des données depuis {input_file}...")
    try:
        df_new = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {input_file} n'existe pas.")
        return

    # 2. Nettoyage
    print("2. Nettoyage et préparation des données...")
    df_processed = clean_and_preprocess(df_new)
    
    # 3. Aligner les colonnes avec celles du modèle
    model_columns = load_columns()
    df_final = df_processed.reindex(columns=model_columns, fill_value=0)
    
    # 4. Charger le modèle et prédire
    print("3. Chargement du modèle et génération des prédictions...")
    model = load_model()
    predictions = model.predict(df_final)
    
    # 5. Ajouter les prédictions au fichier d'origine
    df_new['Churn_Prediction'] = predictions
    df_new['Churn_Risk'] = df_new['Churn_Prediction'].map({0: 'Fidèle', 1: 'Risque de départ'})
    
    # 6. Sauvegarder le résultat
    print(f"4. Sauvegarde des résultats dans {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_new.to_csv(output_file, index=False)
    print("✅ Prédictions terminées avec succès !")

if __name__ == "__main__":
    # Fichiers par défaut pour tester
    # On simule un fichier de nouveaux clients (tu peux utiliser ton fichier dataset.csv pour tester)
    FICHIER_ENTREE = '../data/raw/retail_customers_COMPLETE_CATEGORICAL.csv' 
    FICHIER_SORTIE = '../data/processed/predictions_marketing.csv'
    
    predict_batch(FICHIER_ENTREE, FICHIER_SORTIE)