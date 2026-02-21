import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ===============================
# 1. Charger données prétraitées
# ===============================
X_train = pd.read_csv("data/train_test/X_train.csv")
X_test = pd.read_csv("data/train_test/X_test.csv")
y_train = pd.read_csv("data/train_test/y_train.csv")
y_test = pd.read_csv("data/train_test/y_test.csv")

# ===============================
# 2. Créer modèle
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# ===============================
# 3. Entraîner
# ===============================
model.fit(X_train, y_train.values.ravel())

# ===============================
# 4. Prédictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 5. Évaluation
# ===============================
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report :\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix :\n")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 6. Sauvegarde modèle
# ===============================
joblib.dump(model, "models/model.pkl")

print("\n✅ Modèle entraîné et sauvegardé !")