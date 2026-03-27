import pandas as pd
import numpy as np
from scipy import signal

# Cargar datos
df = pd.read_csv('muscle_fatigue.csv')

# Limpiar nombres de columnas
df.columns = df.columns.str.replace(' ', '_')

# Preprocesar target (Punto 1a)
df['Target'] = df['Target'].replace(2, 1)

# Frecuencia de muestreo 
intervalo = df['Time'].diff().median()
SAMPLING_RATE = int(1 / intervalo)
WINDOW_SIZE = SAMPLING_RATE

print(f"Frecuencia: {SAMPLING_RATE} Hz")

# Canales EMG 
emg_channels = [
    'Right_Rectus_femoris',
    'Left_Gluteus_maximus',
    'Left_Gastrocnemius_medialis',
    'Left_Semitendinosus',
    'Left_Biceps_femoris_caput_longus',
    'Right_Vastus_medialis',
    'Right_Tibialis_anterior',
    'Left_Gastrocnemius_lateralis'
]

# Feature Engineering 
def extract_features(window):
    features = {}

    for channel in emg_channels:
        sig = window[channel].values

        # ── Tiempo ──
        features[f'{channel}_rms'] = np.sqrt(np.mean(sig**2))
        features[f'{channel}_var'] = np.var(sig)
        features[f'{channel}_zc'] = np.sum((sig[:-1] * sig[1:]) < 0)
        features[f'{channel}_slope'] = np.mean(np.diff(sig))

        # ── Frecuencia ──
        freqs, psd = signal.welch(sig, fs=SAMPLING_RATE)
        cumulative = np.cumsum(psd)
        features[f'{channel}_freq_median'] = freqs[
            np.searchsorted(cumulative, cumulative[-1] / 2)
        ]

        if np.sum(psd) == 0:
            features[f'{channel}_freq_mean'] = 0
        else:
            features[f'{channel}_freq_mean'] = np.sum(freqs * psd) / np.sum(psd)

        features[f'{channel}_psd_total'] = np.sum(psd)

    return features

# Ventaneo
rows = []

for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
    window = df.iloc[start : start + WINDOW_SIZE]

    feats = extract_features(window)

    feats['Target'] = window['Target'].mode()[0]

    rows.append(feats)

# Nuevo dataset
df_features = pd.DataFrame(rows)

print(f"Shape: {df_features.shape}")
print(f"Features: {df_features.head()}")
print(f"Description: {df_features.describe()}")
print(df_features['Target'].value_counts())


df_features.to_csv('muscle_fatigue_features.csv', index=False)

print("✓ Features guardadas")