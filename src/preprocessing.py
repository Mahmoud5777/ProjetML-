import pandas as pd

def clean_and_preprocess(df):
    """
    Fonction principale pour nettoyer et préparer les données brutes.
    Utilisée à la fois pour l'entraînement et pour les nouvelles prédictions.
    """
    # Travailler sur une copie pour ne pas modifier l'original
    df_clean = df.copy()

    # 1. Parsing des dates
    if 'RegistrationDate' in df_clean.columns:
        # CORRECTION : Ajout de format='mixed' pour gérer les différents formats sans avertissement
        df_clean['RegistrationDate'] = pd.to_datetime(df_clean['RegistrationDate'], format='mixed', errors='coerce')
        df_clean['RegYear'] = df_clean['RegistrationDate'].dt.year
        df_clean['RegMonth'] = df_clean['RegistrationDate'].dt.month
        df_clean.drop('RegistrationDate', axis=1, inplace=True)

    # 2. Suppression des colonnes inutiles ou constantes
    cols_to_drop = ['NewsletterSubscribed', 'LastLoginIP', 'CustomerID']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    if existing_cols_to_drop:
        df_clean.drop(columns=existing_cols_to_drop, inplace=True)

    # 3. Traitement des valeurs aberrantes et manquantes de base
    if 'Age' in df_clean.columns:
        # CORRECTION : Remplacement direct sans utiliser inplace=True
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
        
    if 'SupportTickets' in df_clean.columns:
        df_clean['SupportTickets'] = df_clean['SupportTickets'].replace([-1, 999], 0)
        # CORRECTION : Remplacement direct sans utiliser inplace=True
        df_clean['SupportTickets'] = df_clean['SupportTickets'].fillna(0)

    # 4. Feature Engineering
    if 'MonetaryTotal' in df_clean.columns and 'Frequency' in df_clean.columns:
        df_clean['AvgBasketValue'] = df_clean['MonetaryTotal'] / df_clean['Frequency'].replace(0, 1)

    # 5. Encodage One-Hot des variables catégorielles (Texte -> Nombres)
    if 'Churn' in df_clean.columns:
        y = df_clean['Churn']
        X = df_clean.drop('Churn', axis=1)
        X = pd.get_dummies(X, drop_first=True)
        X = X.astype(float) # S'assurer que tout est numérique
        X['Churn'] = y
        df_clean = X
    else:
        # Cas d'une prédiction sur un nouveau client
        df_clean = pd.get_dummies(df_clean, drop_first=True)
        df_clean = df_clean.astype(float)

    return df_clean

if __name__ == "__main__":
    print("Le module preprocessing est prêt à être importé !")