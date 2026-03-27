import pandas as pd

# Cargar desde el CSV local que ya se guardó antes
df = pd.read_csv('muscle_fatigue.csv')

'''
# Verificar
print(f"Shape: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nValores del Target:")
print(df['Target'].value_counts())
print(f"\nPrimeras filas:")
df.head()
'''

# Punto 1a: Preprocesar el target
# Convertir etiqueta 2 → 1 (desgaste muscular)
df['Target'] = df['Target'].replace(2, 1)

# Para verificar
print("Valores del Target después del preprocesamiento:")
print(df['Target'].value_counts())

# Punto 1b: Clasificación de variables
print("\nTipos de datos por columna:")
print(df.dtypes)

# Para verificar los valores del dataset
print("\nEstadísticos descriptivos:")
print(df.describe())