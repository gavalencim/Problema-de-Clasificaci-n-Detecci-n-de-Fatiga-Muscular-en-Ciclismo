import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 4.1. Cargar dataset nuevo
df = pd.read_csv('muscle_fatigue_features.csv')

print("Shape del dataset:", df.shape)

# 4.2. Verificar valores nulo
nulls = df.isnull().sum().sum()
print("\nValores nulos totales:", nulls)

if nulls > 0:
    print("⚠️ Eliminando valores nulos...")
    df = df.dropna()
else:
    print("✓ No hay valores nulos")

# 4.3. Separar variables
X = df.drop('Target', axis=1)
y = df['Target']

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

# 4.4. División train / val / test
# 70% train, 15% val, 15% test

# Primero separar test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Luego separar validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.176,  # ≈ 15% del total
    random_state=42,
    stratify=y_train_val
)

print("\nTamaños de los conjuntos:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# 4.5. Balance de clases
print("\nDistribución de clases:")

print("\nTrain:")
print(y_train.value_counts(normalize=True))

print("\nValidation:")
print(y_val.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts(normalize=True))

# 4.6. Escalado
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Datos estandarizados correctamente")

# 4.7. Ejemplo de pipeline (para usar en modelos)
# (No se entrena aquí, solo se deja listo para el punto 5)

from sklearn.neighbors import KNeighborsClassifier

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

print("\n✓ Pipeline de ejemplo creado (kNN)")

# 4.8. Resumen final
print("\nResumen final:")
print("X_train:", X_train_scaled.shape)
print("X_val:", X_val_scaled.shape)
print("X_test:", X_test_scaled.shape)

# 4.9. Guardar datasets

# Train
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

# Validation
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False)

# Test
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
