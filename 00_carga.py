# Librerías necesarias
from datasets import load_dataset
import pandas as pd

# Cargar el dataset desde HuggingFace
dataset = load_dataset("YominE/Muscle_Fatigue_Cycling")

# Convertir a DataFrame de pandas para trabajarlo más fácil
df = pd.DataFrame(dataset['train'])

# Convertir a CSV para poder acceder más facilmente despues
df.to_csv('muscle_fatigue.csv', index=False)
print("✓ Dataset guardado como muscle_fatigue.csv")

