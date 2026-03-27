import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar datos
df = pd.read_csv('muscle_fatigue_features.csv')
df_original = pd.read_csv('muscle_fatigue.csv')

# Graficar 2 segundos de una señal
plt.figure(figsize=(12,4))
plt.plot(df_original['Time'][:2000], df_original['Right Rectus femoris'][:2000])
plt.title('Señal EMG - Right Rectus Femoris')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

# Distribuciones
features_to_plot = [
    'Right_Rectus_femoris_rms',
    'Right_Rectus_femoris_var',
    'Right_Rectus_femoris_freq_mean',
    'Right_Rectus_femoris_psd_total'
]

for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribución de {feature}')
    plt.show()

# Estadísiticas descriptivas
df.describe()

# Correlacion
plt.figure(figsize=(12,10))
corr = df.corr()

sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Matriz de correlación')
plt.show()

# Relacion entre Features y Target
features_to_compare = [
    'Right_Rectus_femoris_rms',
    'Right_Rectus_femoris_freq_mean',
    'Right_Rectus_femoris_psd_total'
]

for feature in features_to_compare:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Target', y=feature, data=df)
    plt.title(f'{feature} vs Target')
    plt.show()

# Balance de clases
sns.countplot(x='Target', data=df)
plt.title('Distribución del Target')
plt.show()

print(df['Target'].value_counts(normalize=True))    

