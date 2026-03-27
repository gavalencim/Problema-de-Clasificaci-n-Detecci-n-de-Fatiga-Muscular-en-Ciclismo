# =========================================================
# 7. PRUEBA CON MUESTRA ARTIFICIAL
# =========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

# 7.1. Cargar datos
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()

X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv').squeeze()

# 7.2. Entrenar modelo final
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = pd.concat([y_train, y_val], axis=0)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train_full, y_train_full)

# 7.3. Generar muestra artificial
# Usar valores promedio + ruido pequeño

mean_values = X_train.mean()
std_values = X_train.std()

sample = mean_values + np.random.normal(0, std_values * 0.1)

sample = pd.DataFrame([sample])

print("\nMuestra generada:")
print(sample.head())

# 7.4. Predicción
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("\n===== RESULTADO =====")

if prediction == 0:
    print("Clasificación: Condición normal")
else:
    print("Clasificación: Fatiga muscular")

print(f"Probabilidades: {probability}")