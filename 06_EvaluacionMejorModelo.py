# =========================================================
# 6. EVALUACIÓN FINAL DEL MEJOR MODELO
# =========================================================

# ── IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# 6.1. Cargar datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()

X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv').squeeze()

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# 6.2. Unir train + val
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = pd.concat([y_train, y_val], axis=0)

print("Nuevo train:", X_train_full.shape)

# 6.3. Entrenar modelo final
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train_full, y_train_full)

# 6.4. Predicciones
y_pred = model.predict(X_test)

# 6.5. Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MÉTRICAS FINALES =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 6.6. Matriz de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# 6.7. Análisis con boxplots
# Agregar predicciones al dataset de test
df_test = X_test.copy()
df_test['Target_real'] = y_test
df_test['Target_pred'] = y_pred

# Ejemplo con algunas features importantes
features_to_plot = [
    'Right_Rectus_femoris_rms',
    'Right_Rectus_femoris_psd_total',
    'Right_Rectus_femoris_freq_mean'
]

for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Target_pred', y=feature, data=df_test)
    plt.title(f'{feature} según clasificación del modelo')
    plt.show()