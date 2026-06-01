#!/usr/bin/env python3
# =============================================================================
# QSVM 5 QUBIT — BAF NeurIPS 2022
# Pipeline: preprocessing -> MI -> PCA(4) + top-MI feature -> ZZFeatureMap(5 qubit)
# Requisiti:
# pip install qiskit qiskit-aer scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 78)
print("QSVM 5 QUBIT — PCA(4) + TOP FEATURE MI")
print("=" * 78)

# -----------------------------------------------------------------------------
# BLOCCO 1 — Caricamento dataset
# -----------------------------------------------------------------------------
print("=" * 70)
print("BLOCCO 1 — CARICAMENTO DATASET")
print("=" * 70)
df = pd.read_csv("Base.csv")
print(f"Shape originale: {df.shape}")
print(f"Distribuzione classi: {df['fraud_bool'].value_counts().to_dict()}")
print(f"Percentuale frodi: {df['fraud_bool'].mean()*100:.2f}%")

# -----------------------------------------------------------------------------
# BLOCCO 2 — Preprocessing
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 2 — PREPROCESSING")
print("=" * 70)
# 2a. Drop feature con data leakage e variabile temporale
DROP = ["device_fraud_count", "month"]
df.drop(columns=DROP, inplace=True, errors="ignore")
print(f"[DROP] colonne rimosse: {DROP}")

# 2b. Encoding variabili categoriche → LabelEncoder
# Identifica AUTOMATICAMENTE tutte le colonne object/string rimaste
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "fraud_bool"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 2b-bis. Forza conversione numerica su tutte le colonne rimanenti
# (cattura valori come 'BC', 'CA', ecc. rimasti in colonne non-object)
for col in df.columns:
    if col == "fraud_bool":
        continue
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# 2c. Imputazione valori mancanti codificati come -1
missing_cols = [
    "prev_address_months_count",
    "current_address_months_count",
    "intended_balcon_amount",
    "bank_months_count"
]
for col in missing_cols:
    if col in df.columns:
        med = df.loc[df[col] != -1, col].median()
        df[col] = df[col].replace(-1, med)
        print(f"[IMPUTE] {col}: -1 -> mediana {med:.3f}")

# 2d. Separazione X / y
X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

# 2e. StandardScaler (µ=0, σ=1) — obbligatorio prima di PCA e MI
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(f"[SCALE] StandardScaler su {X.shape[1]} feature")

# -----------------------------------------------------------------------------
# BLOCCO 3 — Mutual Information
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 3 — FEATURE SELECTION: MUTUAL INFORMATION")
print("=" * 70)
mi = mutual_info_classif(X_scaled, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
top_mi_feature = mi_series.index[0]
top_mi_score = mi_series.iloc[0]

print("\nTop 10 feature per Mutual Information:")
print(mi_series.head(10).to_string())
print(f"\n[TOP MI] Quinta feature selezionata: {top_mi_feature} (MI={top_mi_score:.6f})")

# Plot MI scores
fig, ax = plt.subplots(figsize=(10, 5))
mi_series.head(15).plot(kind="barh", ax=ax, color="#01696f")
ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Importance — Mutual Information vs fraud_bool")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("mi_scores_5q.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# BLOCCO 4 — PCA a 4 componenti + quinta feature MI
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 4 — FEATURE PROJECTION: PCA → 4 COMPONENTI")
print("=" * 70)
# Definisci quanti qubit useremo in totale per la feature map
# 4 saranno per PCA + 1 per la feature MI
N_PCA = 4
N_QUBITS = 5

# Esegui PCA solo sulle feature numeriche standardizzate
pca = PCA(n_components=N_PCA, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Calcola varianza spiegata dalle componenti PCA
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

mi_feature_values = X_scaled_df[[top_mi_feature]].values
X_5 = np.hstack([X_pca, mi_feature_values])

print("\nVarianza spiegata dalle 4 componenti PCA:")
for i, (ev, cum) in enumerate(zip(explained, cumulative), start=1):
    print(f" PC{i}: {ev*100:.2f}% (cumulativo: {cum*100:.2f}%)")
print(f"[INFO] Le prime 4 dimensioni sono PCA; il 5° qubit codifica direttamente '{top_mi_feature}'.")
print("[INFO] Questo non aumenta la varianza PCA, ma preserva esplicitamente la feature più informativa.")

# Plot varianza spiegata PCA + feature MI
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [f"PC{i}" for i in range(1, N_PCA + 1)]
ax.bar(labels, explained * 100, color="#01696f", alpha=0.85)
ax.plot(labels, cumulative * 100, "o--", color="#EF553B", lw=2, label="Cumulativo PCA")
for i, c in enumerate(cumulative):
    ax.annotate(f"{c*100:.1f}%", (i, c*100 + 0.6), ha="center", fontsize=10)
ax.axhline(cumulative[-1] * 100, color="#636EFA", linestyle=":", lw=2,
           label=f"PCA(4) = {cumulative[-1]*100:.1f}%")
ax.set_ylabel("Varianza spiegata (%)")
ax.set_title(f"PCA(4) + top MI feature: {top_mi_feature}")
ax.legend()
plt.tight_layout()
plt.savefig("pca_plus_mi_5q.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# BLOCCO 5 — Bilanciamento classi
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 5 — BILANCIAMENTO CLASSI (100 per classe)")
print("=" * 70)
N_CLASS = 150
feature_cols_5 = [f"PC{i}" for i in range(1, N_PCA + 1)] + [top_mi_feature]

# Crea un DataFrame con le 5 feature e la label
df_5 = pd.DataFrame(X_5, columns=feature_cols_5)
df_5["fraud_bool"] = y.values
legit = df_5[df_5["fraud_bool"] == 0]
fraud = df_5[df_5["fraud_bool"] == 1]
# Undersampling del dataset per il bilanciamento delle classi:
# selezioniamo 150 campioni da ciascuna classe in modo casuale
legit_down = resample(legit, n_samples=N_CLASS, random_state=42, replace=False)
fraud_bal = resample(fraud, n_samples=N_CLASS, random_state=42, replace=(len(fraud) < N_CLASS))
# Combiniamo i campioni selezionati e mescoliamoli
bal = pd.concat([legit_down, fraud_bal]).sample(frac=1, random_state=42)
X_bal = bal[feature_cols_5].values
y_bal = bal["fraud_bool"].values
print(f"\nDataset bilanciato: {bal.shape}")
print(f"Classi bilanciate: {bal['fraud_bool'].value_counts().to_dict()}")

# -----------------------------------------------------------------------------
# BLOCCO 6 — Scaling angolare [-pi, pi] per ZZFEATUREMAP
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 6 — SCALING ANGOLARE [-π, π] per ZZFEATUREMAP")
print("=" * 70)
angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_angle = angle_scaler.fit_transform(X_bal)
print(f"Range finale input ZZFeatureMap: [{X_angle.min():.4f}, {X_angle.max():.4f}] rad")

# -----------------------------------------------------------------------------
# BLOCCO 7 — Train/Test split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_angle, y_bal, test_size=0.25, random_state=42, stratify=y_bal
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# -----------------------------------------------------------------------------
# BLOCCO 8 — Kernel quantistico a 5 qubit
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 8 — QUANTUM KERNEL: ZZFeatureMap + FidelityQuantumKernel")
print("=" * 70)
from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
import itertools

# ── 8a. ZZFeatureMap ─────────────────────────────────────────────────────────
feature_map = ZZFeatureMap(
    feature_dimension=N_QUBITS,
    reps=1,
    entanglement="full"
)

print("\nZZFeatureMap 5-qubit creata.")
feature_map_txt = feature_map.decompose().draw(output="text", fold=-1)
feature_map_txt = str(feature_map_txt)
print(feature_map_txt)
with open("zzfeaturemap.txt", "w", encoding="utf-8") as f:
    f.write(feature_map_txt)
print("[SAVE] Salvato zzfeaturemap.txt")

# ── 8b. Backend AerSimulator ─────────────────────────────────────────────────

try:
    backend = AerSimulator(method="statevector", device="GPU")
    backend_target = "GPU"
except Exception:
    backend = AerSimulator(method="statevector")
    backend_target = "CPU"
print(f"[BACKEND] AerSimulator in uso: {backend_target}")

N_SHOTS = 8192 # shots per stima della fidelity


def compute_kernel_entry(xi, xj, fmap, backend, shots=N_SHOTS):
    """
    Calcola K(xi, xj) = P(|00...0>) tramite il circuito compute-uncompute.
    Pipeline manuale:
      1. Assegna xi alla feature_map  → U(xi)
      2. Assegna xj alla feature_map  → U(xj), poi prende U†(xj)
      3. Compone U(xi) · U†(xj) e aggiunge misure
      4. Esegue su AerSimulator e legge P(000...0)
    """
    params = fmap.parameters
    circ_xi = fmap.assign_parameters(dict(zip(params, xi)))
    circ_xj_inv = fmap.assign_parameters(dict(zip(params, xj))).inverse()

    n = fmap.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(circ_xi, inplace=True)
    qc.compose(circ_xj_inv, inplace=True)
    qc.measure(range(n), range(n))

    qc_t = transpile(qc, backend, optimization_level=0)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    return counts.get("0" * n, 0) / shots


def compute_kernel_matrix(X_a, X_b, fmap, backend, shots=N_SHOTS, symmetric=False):
    """
    Calcola la matrice kernel K[i,j] = compute_kernel_entry(X_a[i], X_b[j]).
    Se symmetric=True sfrutta K[i,j]=K[j,i] dimezzando le computazioni.
    """
    n_a, n_b = len(X_a), len(X_b)
    K = np.zeros((n_a, n_b))

    if symmetric:
        pairs = [(i, j) for i in range(n_a) for j in range(i, n_b)]
        total = len(pairs)
        for idx, (i, j) in enumerate(pairs):
            if idx % 100 == 0:
                print(f" Progresso train: {idx}/{total} ({idx/total*100:.1f}%)", end="\r", flush=True)
            val = compute_kernel_entry(X_a[i], X_b[j], fmap, backend, shots)
            K[i, j] = val
            K[j, i] = val
        print(f" Progresso train: {total}/{total} (100.0%)")
    else:
        total = n_a * n_b
        for idx, (i, j) in enumerate(itertools.product(range(n_a), range(n_b))):
            if idx % 100 == 0:
                print(f" Progresso test: {idx}/{total} ({idx/total*100:.1f}%)", end="\r", flush=True)
            K[i, j] = compute_kernel_entry(X_a[i], X_b[j], fmap, backend, shots)
        print(f" Progresso test: {total}/{total} (100.0%)")
    return K

# ── 8c. Calcolo K_train ───────────────────────────────────────────────────────
print(f"\nCalcolo kernel matrix train ({X_train.shape[0]}x{X_train.shape[0]})...")
print(f"  N° coppie uniche: {X_train.shape[0]*(X_train.shape[0]+1)//2}")
print(f"  Shots per coppia: {N_SHOTS}")
print(f"  Stima tempo      : ~{X_train.shape[0]*(X_train.shape[0]+1)//2 * 0.008 / 60:.0f}-"
      f"{X_train.shape[0]*(X_train.shape[0]+1)//2 * 0.02 / 60:.0f} min")
K_train = compute_kernel_matrix(X_train, X_train, feature_map, backend, symmetric=True)
print(f"  K_train shape: {K_train.shape}")

# ── 8d. Calcolo K_test ────────────────────────────────────────────────────────
print(f"Calcolo K_test ({X_test.shape[0]}x{X_train.shape[0]})...")
K_test = compute_kernel_matrix(X_test, X_train, feature_map, backend, symmetric=False)
print(f"  K_test shape: {K_test.shape}")

# ── 8e. Verifica proprietà ────────────────────────────────────────────────────
print(f"Diagonale media K_train: {np.diag(K_train).mean():.4f}")
print(f"Errore massimo di simmetria: {np.max(np.abs(K_train - K_train.T)):.6f}")
print(f"Range K_train: [{K_train.min():.4f}, {K_train.max():.4f}]")

# -----------------------------------------------------------------------------
# BLOCCO 9 — QSVM + benchmark classico
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 9 — SVM QUANTUM (kernel='precomputed') e confronto con SVM CLASSICA (Linear e RBF)")
print("=" * 70)
qsvm = SVC(kernel="precomputed", C=1.0, probability=True, random_state=42)
qsvm.fit(K_train, y_train)
y_pred_q = qsvm.predict(K_test)
y_prob_q = qsvm.predict_proba(K_test)[:, 1]

svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm_lin.fit(X_train, y_train)
yp_l = svm_lin.predict(X_test)
yp_l_prob = svm_lin.predict_proba(X_test)[:, 1]

svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm_rbf.fit(X_train, y_train)
yp_r = svm_rbf.predict(X_test)
yp_r_prob = svm_rbf.predict_proba(X_test)[:, 1]

results = {
    "QSVM (5 qubit: PCA4 + MI1)": {
        "Accuracy": accuracy_score(y_test, y_pred_q),
        "Precision": precision_score(y_test, y_pred_q, zero_division=0),
        "Recall": recall_score(y_test, y_pred_q, zero_division=0),
        "F1": f1_score(y_test, y_pred_q, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob_q),
    },
    "SVM Lineare": {
        "Accuracy": accuracy_score(y_test, yp_l),
        "Precision": precision_score(y_test, yp_l, zero_division=0),
        "Recall": recall_score(y_test, yp_l, zero_division=0),
        "F1": f1_score(y_test, yp_l, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, yp_l_prob),
    },
    "SVM RBF": {
        "Accuracy": accuracy_score(y_test, yp_r),
        "Precision": precision_score(y_test, yp_r, zero_division=0),
        "Recall": recall_score(y_test, yp_r, zero_division=0),
        "F1": f1_score(y_test, yp_r, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, yp_r_prob),
    }
}

df_res = pd.DataFrame(results).T
print("\nTABELLA RISULTATI")
print(df_res.round(4).to_string())

# -----------------------------------------------------------------------------
# BLOCCO 10 — Grafici
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(K_train, cmap="YlGnBu", ax=ax,
            cbar_kws={"label": "Fidelity K(xi,xj)"},
            xticklabels=False, yticklabels=False)
ax.set_title(f"Kernel quantistico — 5 qubit\nTrain set ({X_train.shape[0]} campioni)")
plt.tight_layout()
plt.savefig("kernel_matrix_5q.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["Accuracy", "F1", "ROC-AUC"]
colors_bar = ["#01696f", "#636EFA", "#EF553B"]
for ax, metric in zip(axes, metrics):
    vals = df_res[metric]
    bars = ax.bar(vals.index, vals.values, color=colors_bar, width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=18)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
fig.suptitle("QSVM 5 qubit vs SVM classiche — PCA(4)+MI(1)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("benchmark_metrics_5q.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [
    ("QSVM 5q", y_pred_q),
    ("SVM Lineare", yp_l),
    ("SVM RBF", yp_r)
]
for ax, (name, preds) in zip(axes, models):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legittimo", "Frode"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12, fontweight="bold")
fig.suptitle("Confusion matrices — setup 5 qubit", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrices_5q.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# BLOCCO 11 — Salvataggio
# -----------------------------------------------------------------------------
df_res.to_csv("benchmark_results_5q.csv")
np.save("K_train_5q.npy", K_train)
np.save("K_test_5q.npy", K_test)
np.save("X_train_5q.npy", X_train)
np.save("X_test_5q.npy", X_test)
np.save("y_train_5q.npy", y_train)
np.save("y_test_5q.npy", y_test)

with open("readme_5q.txt", "w", encoding="utf-8") as f:
    f.write(
        "QSVM 5 qubit\n"
        f"Input finale: PC1, PC2, PC3, PC4, {top_mi_feature}\n"
        f"Top feature MI: {top_mi_feature} (score={top_mi_score:.6f})\n"
        f"Varianza cumulativa PCA(4): {cumulative[-1]*100:.2f}%\n"
        "Nota: la 5a feature non aumenta la varianza PCA, ma preserva informazione supervisionata utile.\n"
        "Per usare GPU NVIDIA con Qiskit Aer, installa qiskit-aer-gpu e abilita device='GPU'.\n"
    )

print("\n" + "=" * 78)
print("PIPELINE 5 QUBIT COMPLETATA")
print("Output generati:")
print("- mi_scores_5q.png")
print("- pca_plus_mi_5q.png")
print("- kernel_matrix_5q.png")
print("- benchmark_metrics_5q.png")
print("- confusion_matrices_5q.png")
print("- benchmark_results_5q.csv")
print("- K_train_5q.npy / K_test_5q.npy")
print("- X_train_5q.npy / X_test_5q.npy / y_train_5q.npy / y_test_5q.npy")
print("- readme_5q.txt")
print("=" * 78)
