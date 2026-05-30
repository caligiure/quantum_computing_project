# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE — BAF Dataset → QSVM con ZZFeatureMap (4 qubit)
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ── 1. Caricamento ────────────────────────────────────────────────────────────
df = pd.read_csv('Base.csv')

# ── 2. Drop colonne con data leakage e temporali ──────────────────────────────
df.drop(columns=['device_fraud_count', 'month'], inplace=True, errors='ignore')

# ── 3. Encoding variabili categoriche ─────────────────────────────────────────
CAT_COLS = ['payment_type', 'source', 'device_os', 'employment_status']
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ── 4. Gestione valori mancanti codificati come -1 ────────────────────────────
MISSING_COLS = ['prev_address_months_count', 'current_address_months_count',
                'intended_balcon_amount', 'bank_months_count']
for col in MISSING_COLS:
    median_val = df.loc[df[col] != -1, col].median()
    df[col] = df[col].replace(-1, median_val)

# ── 5. Separazione X / y ──────────────────────────────────────────────────────
X = df.drop(columns=['fraud_bool'])
y = df['fraud_bool']

# ── 6. StandardScaler ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 7. Feature Selection (Mutual Information) — opzionale, diagnostico ────────
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Top 10 feature per MI:\\n", mi_series.head(10))

# ── 8. PCA → 4 componenti (= 4 qubit) ────────────────────────────────────────
N_QUBITS = 4
pca = PCA(n_components=N_QUBITS, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Varianza spiegata:", np.cumsum(pca.explained_variance_ratio_))

# ── 9. Bilanciamento dataset (100 per classe) ─────────────────────────────────
N_PER_CLASS = 100
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(N_QUBITS)])
X_pca_df['fraud_bool'] = y.values

df_legit = X_pca_df[X_pca_df['fraud_bool'] == 0]
df_fraud  = X_pca_df[X_pca_df['fraud_bool'] == 1]

df_legit_down     = resample(df_legit, n_samples=N_PER_CLASS, random_state=42, replace=False)
n_fraud           = len(df_fraud)
df_fraud_balanced = resample(df_fraud, n_samples=N_PER_CLASS, random_state=42,
                             replace=(n_fraud < N_PER_CLASS))

df_balanced = pd.concat([df_legit_down, df_fraud_balanced]).sample(frac=1, random_state=42)

X_balanced = df_balanced[[f'PC{i+1}' for i in range(N_QUBITS)]].values
y_balanced = df_balanced['fraud_bool'].values

# ── 10. Scaling angolare in [-π, π] per ZZFeatureMap ─────────────────────────
angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_angle = angle_scaler.fit_transform(X_balanced)

# ── 11. Train / Test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_angle, y_balanced, test_size=0.25, random_state=42, stratify=y_balanced
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── OUTPUT PRONTO PER QISKIT ──────────────────────────────────────────────────
# X_train → (150, 4)  valori in [-π, π]
# X_test  → (50, 4)   valori in [-π, π]
# y_train, y_test → array binari {0, 1}
#
# Utilizzo con ZZFeatureMap:
# from qiskit.circuit.library import ZZFeatureMap
# feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')