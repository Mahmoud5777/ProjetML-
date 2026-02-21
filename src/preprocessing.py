import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ===============================
# 1. Charger données
# ===============================
df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

print("Colonnes détectées :")
print(df.columns)

# ===============================
# 2. Supprimer colonnes inutiles
# ===============================
df.drop(columns=["NewsletterSubscribed", "LastLoginIP"],
        errors="ignore", inplace=True)

# ===============================
# 3. Séparer target
# ===============================
y = df["Churn"]
X = df.drop("Churn", axis=1)

# ===============================
# 4. Identifier colonnes
# ===============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Colonnes numériques :", numeric_features)
print("Colonnes catégorielles :", categorical_features)

# ===============================
# 5. Pipelines
# ===============================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ===============================
# 6. Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 7. Appliquer preprocessing
# ===============================
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# ===============================
# 8. Sauvegarde
# ===============================
joblib.dump(preprocessor, "models/preprocessor.pkl")

pd.DataFrame(X_train).to_csv("data/train_test/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/train_test/X_test.csv", index=False)
y_train.to_csv("data/train_test/y_train.csv", index=False)
y_test.to_csv("data/train_test/y_test.csv", index=False)

print("✅ Preprocessing terminé avec succès !")