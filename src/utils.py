import joblib
import os

def load_model(model_path='../models/production_model.joblib'):
    """Charge le modèle ML sauvegardé."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} est introuvable. Entraînez-le d'abord.")
    return joblib.load(model_path)

def load_columns(columns_path='../models/model_columns.joblib'):
    """Charge la liste des colonnes attendues par le modèle."""
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"Le fichier {columns_path} est introuvable.")
    return joblib.load(columns_path)