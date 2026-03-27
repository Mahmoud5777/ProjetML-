import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# On importe ta fameuse "boîte à outils" !
from preprocessing import clean_and_preprocess

def train():
    print("=== DÉMARRAGE DE L'ENTRAÎNEMENT ===")
    
    # 1. Charger les données brutes
    print("1. Chargement des données brutes...")
    df_raw = pd.read_csv('../data/raw/retail_customers_COMPLETE_CATEGORICAL.csv') 
    
    # 2. Nettoyage automatique grâce à ton module
    print("2. Nettoyage et Feature Engineering...")
    df_clean = clean_and_preprocess(df_raw)
    
    # 3. Préparation pour le Machine Learning
    print("3. Séparation Train/Test...")
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Entraînement du modèle (On prend les paramètres optimisés trouvés précédemment)
    print("4. Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. Évaluation rapide
    print("5. Évaluation sur les données de Test :")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 6. Sauvegarde du modèle final
    print("6. Sauvegarde du modèle...")
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/production_model.joblib')
    
    joblib.dump(list(X.columns), '../models/model_columns.joblib')
    
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ! Modèle prêt pour la production.")

if __name__ == "__main__":
    train()