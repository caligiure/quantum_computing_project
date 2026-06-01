#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# QSVM COMPLETA — Bank Account Fraud (BAF NeurIPS 2022)
# Pipeline: Preprocessing → Quantum Kernel → SVM → Benchmarking classico
# Requisiti: pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn
#             pandas numpy matplotlib seaborn imbalanced-learn
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("[INFO] Per GPU NVIDIA serve qiskit-aer-gpu e CUDA configurato correttamente.")

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 1 ── CARICAMENTO E ISPEZIONE DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("BLOCCO 1 — CARICAMENTO DATASET")
print("=" * 70)

df = pd.read_csv("Base.csv")        # ← assicurati che Base.csv sia nella cwd

print(f"Shape originale : {df.shape}")
print(f"Classi          : {df['fraud_bool'].value_counts().to_dict()}")
print(f"% frodi         : {df['fraud_bool'].mean()*100:.2f}%")
print(f"Missing totali  : {(df == -1).sum().sum()} (codificati come -1)")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 2 ── PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 2 — PREPROCESSING")
print("=" * 70)

# 2a. Drop feature con data leakage e variabile temporale
DROP = ["device_fraud_count", "month"]
df.drop(columns=DROP, inplace=True, errors="ignore")
print(f"[DROP] Rimosse: {DROP}")

# 2b. Encoding variabili categoriche → LabelEncoder
# Identifica AUTOMATICAMENTE tutte le colonne object/string rimaste
# Questo cattura qualunque colonna stringa, non solo le 4 note
CAT = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Rimuovi il target se per caso fosse stringa
CAT = [c for c in CAT if c != "fraud_bool"]
for col in CAT:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print(f"[ENCODE] LabelEncoded ({len(CAT)} colonne): {CAT}")

# 2b-bis. Forza conversione numerica su tutte le colonne rimanenti
# (cattura valori come 'BC', 'CA', ecc. rimasti in colonne non-object)
for col in df.columns:
    if col == "fraud_bool":
        continue
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        # colonna ancora stringa → applica LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [ENCODE extra] {col} codificata (trovata in secondo passaggio)")

# 2c. Imputazione valori mancanti codificati come -1
MISSING = ["prev_address_months_count", "current_address_months_count",
           "intended_balcon_amount", "bank_months_count"]
for col in MISSING:
    med = df.loc[df[col] != -1, col].median()
    df[col] = df[col].replace(-1, med)
    print(f"[IMPUTE] {col}: -1 → mediana {med:.1f}")

# 2d. Separazione X / y
X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

# 2e. StandardScaler (µ=0, σ=1) — obbligatorio prima di PCA e MI
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"[SCALE] StandardScaler applicato su {X.shape[1]} feature")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 3 ── FEATURE SELECTION (Mutual Information) — diagnostico
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 3 — FEATURE SELECTION: MUTUAL INFORMATION")
print("=" * 70)

mi = mutual_info_classif(X_scaled, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("Top 10 feature per Mutual Information (I(X;Y)):")
print(mi_series.head(10).to_string())

# Plot MI scores
fig, ax = plt.subplots(figsize=(10, 5))
mi_series.head(15).plot(kind="barh", ax=ax, color="#01696f")
ax.set_xlabel("Mutual Information Score", fontsize=12)
ax.set_title("Feature Importance — Mutual Information vs fraud_bool", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("mi_scores.png", dpi=150)
plt.close()
print("[PLOT] Salvato mi_scores.png")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 4 ── FEATURE PROJECTION (PCA → 4 componenti = 4 qubit)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 4 — FEATURE PROJECTION: PCA → 4 COMPONENTI")
print("=" * 70)

N_QUBITS = 4
pca = PCA(n_components=N_QUBITS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

ev_ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(ev_ratio)
print(f"Varianza spiegata per componente:")
for i, (ev, cum) in enumerate(zip(ev_ratio, cumulative)):
    print(f"  PC{i+1}: {ev*100:.2f}%  (cumulativo: {cum*100:.2f}%)")

# Plot scree
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([f"PC{i+1}" for i in range(N_QUBITS)], ev_ratio*100, color="#01696f", alpha=0.85)
ax.plot([f"PC{i+1}" for i in range(N_QUBITS)], cumulative*100,
        "o--", color="#EF553B", lw=2, label="Cumulativo")
for i, c in enumerate(cumulative):
    ax.annotate(f"{c*100:.1f}%", (i, c*100+0.5), ha="center", fontsize=10)
ax.set_ylabel("Varianza spiegata (%)")
ax.set_title("PCA Scree Plot — 4 componenti per ZZFeatureMap")
ax.legend()
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150)
plt.close()
print("[PLOT] Salvato pca_scree.png")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 5 ── BILANCIAMENTO E CAMPIONAMENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 5 — BILANCIAMENTO CLASSI (100 per classe)")
print("=" * 70)

N_CLASS = 150 

df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(N_QUBITS)])
df_pca["fraud_bool"] = y.values

legit = df_pca[df_pca["fraud_bool"] == 0]
fraud = df_pca[df_pca["fraud_bool"] == 1]

legit_down = resample(legit, n_samples=N_CLASS, random_state=42, replace=False)
n_fraud = len(fraud)
fraud_bal = resample(fraud, n_samples=N_CLASS, random_state=42,
                     replace=(n_fraud < N_CLASS))

bal = pd.concat([legit_down, fraud_bal]).sample(frac=1, random_state=42)
print(f"Dataset bilanciato: {bal.shape}")
print(f"Classi: {bal['fraud_bool'].value_counts().to_dict()}")

X_bal = bal[[f"PC{i+1}" for i in range(N_QUBITS)]].values
y_bal = bal["fraud_bool"].values

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 6 ── SCALING ANGOLARE IN [-π, π] PER ZZFEATUREMAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 6 — SCALING ANGOLARE [-π, π]")
print("=" * 70)

angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_angle = angle_scaler.fit_transform(X_bal)
print(f"Range output: [{X_angle.min():.4f}, {X_angle.max():.4f}] rad")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 7 ── TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 7 — TRAIN/TEST SPLIT (75/25 stratificato)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_angle, y_bal, test_size=0.25, random_state=42, stratify=y_bal
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# Scatter 2D: PC1 vs PC2
fig, ax = plt.subplots(figsize=(7, 6))
mask0 = y_bal == 0; mask1 = y_bal == 1
ax.scatter(X_angle[mask0, 0], X_angle[mask0, 1], alpha=0.6, s=30,
           marker="o", color="#636EFA", label="Legittimo (0)")
ax.scatter(X_angle[mask1, 0], X_angle[mask1, 1], alpha=0.9, s=45,
           marker="X", color="#EF553B", label="Frode (1)")
ax.set_xlabel("PC1 [radianti]"); ax.set_ylabel("PC2 [radianti]")
ax.set_title("PC1 vs PC2 — 200 campioni bilanciati (input QSVM)")
ax.legend(); plt.tight_layout()
plt.savefig("scatter_pc1_pc2.png", dpi=150)
plt.close()
print("[PLOT] Salvato scatter_pc1_pc2.png")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 8 ── QUANTUM KERNEL con QISKIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BLOCCO 8 — QUANTUM KERNEL: ZZFeatureMap + FidelityQuantumKernel")
print("=" * 70)

from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile
from qiskit_aer import AerSimulator
import itertools

# ── 8a. ZZFeatureMap ─────────────────────────────────────────────────────────
feature_map = ZZFeatureMap(
    feature_dimension=N_QUBITS,
    reps=1, # Con reps=2 il circuito è molto profondo → i punti diventano troppo ortogonali (kernel ~0 ovunque). Con reps=1 il kernel è meno "espressivo" ma meno soggetto a concentration, e con pochi qubit spesso funziona meglio.
    entanglement="full" # L'entanglement full connette tutti i qubit tra loro (non solo adiacenti), creando correlazioni più ricche tra le 4 feature. Con soli 4 qubit il costo computazionale è accettabile.
)
print("ZZFeatureMap creata:")
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

N_SHOTS = 8192   # shots per stima della fidelity

def compute_kernel_entry(xi, xj, feature_map, backend, shots=N_SHOTS):
    """
    Calcola K(xi, xj) = P(|00...0>) tramite il circuito compute-uncompute.
    Pipeline manuale:
      1. Assegna xi alla feature_map  → U(xi)
      2. Assegna xj alla feature_map  → U(xj), poi prende U†(xj)
      3. Compone U(xi) · U†(xj) e aggiunge misure
      4. Esegue su AerSimulator e legge P(000...0)
    """
    # Parametri della feature_map
    params = feature_map.parameters

    # Circuito per xi
    circ_xi = feature_map.assign_parameters(dict(zip(params, xi)))

    # Circuito inverso per xj
    circ_xj_inv = feature_map.assign_parameters(dict(zip(params, xj))).inverse()

    # Circuito compute-uncompute
    from qiskit import QuantumCircuit
    n = feature_map.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(circ_xi,     inplace=True)
    qc.compose(circ_xj_inv, inplace=True)
    qc.measure(range(n), range(n))

    # Transpile e run
    qc_t = transpile(qc, backend, optimization_level=0)
    job  = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()

    # P(|00...0>) = fidelity ≈ K(xi, xj)
    zero_state = "0" * n
    return counts.get(zero_state, 0) / shots


def compute_kernel_matrix(X_a, X_b, feature_map, backend,
                          shots=N_SHOTS, symmetric=False):
    """
    Calcola la matrice kernel K[i,j] = compute_kernel_entry(X_a[i], X_b[j]).
    Se symmetric=True sfrutta K[i,j]=K[j,i] dimezzando le computazioni.
    """
    n_a, n_b = len(X_a), len(X_b)
    K = np.zeros((n_a, n_b))

    if symmetric:
        # Solo triangolo superiore + diagonale
        pairs = [(i, j) for i in range(n_a) for j in range(i, n_b)]
        total = len(pairs)
        for idx, (i, j) in enumerate(pairs):
            if idx % 100 == 0:
                print(f"  Progresso: {idx}/{total} coppie ({idx/total*100:.1f}%)",
                      end="\r", flush=True)
            val = compute_kernel_entry(X_a[i], X_b[j], feature_map, backend, shots)
            K[i, j] = val
            K[j, i] = val
        print(f"  Completato: {total}/{total} coppie (100.0%)        ")
    else:
        total = n_a * n_b
        for idx, (i, j) in enumerate(itertools.product(range(n_a), range(n_b))):
            if idx % 100 == 0:
                print(f"  Progresso: {idx}/{total} coppie ({idx/total*100:.1f}%)",
                      end="\r", flush=True)
            K[i, j] = compute_kernel_entry(X_a[i], X_b[j],
                                            feature_map, backend, shots)
        print(f"  Completato: {total}/{total} coppie (100.0%)        ")
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
print(f"\nCalcolo kernel matrix test ({X_test.shape[0]}x{X_train.shape[0]})...")
K_test = compute_kernel_matrix(X_test, X_train, feature_map, backend, symmetric=False)
print(f"  K_test shape: {K_test.shape}")

# ── 8e. Verifica proprietà ────────────────────────────────────────────────────
print(f"  Diagonale media  : {np.diag(K_train).mean():.4f} (atteso: ~1.0)")
print(f"  Simmetria max err: {np.max(np.abs(K_train - K_train.T)):.6f}")
print(f"  Range valori     : [{K_train.min():.4f}, {K_train.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 9 ── ADDESTRAMENTO SVM CON KERNEL PRECOMPUTED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 9 — SVM QUANTUM (kernel='precomputed')")
print("=" * 70)

# K_train viene passato direttamente come matrice di Gram a sklearn SVC
# Il classificatore non "vede" mai i dati originali: solo il kernel matrix
qsvm = SVC(kernel="precomputed", C=1.0, probability=True, random_state=42)
qsvm.fit(K_train, y_train)
print(f"QSVM addestrata. Support vectors: {qsvm.n_support_}")

y_pred_q  = qsvm.predict(K_test)
y_prob_q  = qsvm.predict_proba(K_test)[:, 1]

acc_q  = accuracy_score(y_test, y_pred_q)
prec_q = precision_score(y_test, y_pred_q, zero_division=0)
rec_q  = recall_score(y_test, y_pred_q, zero_division=0)
f1_q   = f1_score(y_test, y_pred_q, zero_division=0)
auc_q  = roc_auc_score(y_test, y_prob_q)

print(f"  Accuracy  : {acc_q:.4f}")
print(f"  Precision : {prec_q:.4f}")
print(f"  Recall    : {rec_q:.4f}")
print(f"  F1-Score  : {f1_q:.4f}")
print(f"  ROC-AUC   : {auc_q:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 10 ── BENCHMARKING: SVM CLASSICA (Linear + RBF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 10 — BENCHMARKING: SVM LINEARE + SVM RBF")
print("=" * 70)

results = {}

# QSVM
results["QSVM (ZZ-Feature Map)"] = {
    "Accuracy": acc_q, "Precision": prec_q,
    "Recall": rec_q, "F1": f1_q, "ROC-AUC": auc_q
}

# SVM Lineare
svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm_lin.fit(X_train, y_train)
yp_l = svm_lin.predict(X_test)
yp_l_prob = svm_lin.predict_proba(X_test)[:, 1]
results["SVM Lineare"] = {
    "Accuracy":  accuracy_score(y_test, yp_l),
    "Precision": precision_score(y_test, yp_l, zero_division=0),
    "Recall":    recall_score(y_test, yp_l, zero_division=0),
    "F1":        f1_score(y_test, yp_l, zero_division=0),
    "ROC-AUC":   roc_auc_score(y_test, yp_l_prob)
}

# SVM RBF
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm_rbf.fit(X_train, y_train)
yp_r = svm_rbf.predict(X_test)
yp_r_prob = svm_rbf.predict_proba(X_test)[:, 1]
results["SVM RBF"] = {
    "Accuracy":  accuracy_score(y_test, yp_r),
    "Precision": precision_score(y_test, yp_r, zero_division=0),
    "Recall":    recall_score(y_test, yp_r, zero_division=0),
    "F1":        f1_score(y_test, yp_r, zero_division=0),
    "ROC-AUC":   roc_auc_score(y_test, yp_r_prob)
}

df_res = pd.DataFrame(results).T
print("\n── TABELLA COMPARATIVA ──────────────────────────────────────────")
print(df_res.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 11 ── VISUALIZZAZIONI DI CONFRONTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 11 — VISUALIZZAZIONI")
print("=" * 70)

# 11a. Heatmap matrice kernel quantistica
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(K_train, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Fidelity K(xi,xj)"},
            xticklabels=False, yticklabels=False)
ax.set_title(f"Matrice Kernel Quantistica (ZZFeatureMap, {N_QUBITS} qubit)\n"
             f"Training set ({X_train.shape[0]} campioni)")
plt.tight_layout(); plt.savefig("kernel_matrix.png", dpi=150); plt.close()
print("[PLOT] Salvato kernel_matrix.png")

# 11b. Confronto metriche — bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["Accuracy", "F1", "ROC-AUC"]
colors_bar = ["#01696f", "#636EFA", "#EF553B"]
for ax, metric in zip(axes, metrics):
    vals = df_res[metric]
    bars = ax.bar(vals.index, vals.values, color=colors_bar, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=10)
fig.suptitle("Confronto QSVM vs SVM Classiche — Dataset BAF NeurIPS 2022",
             fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("benchmark_metrics.png", dpi=150); plt.close()
print("[PLOT] Salvato benchmark_metrics.png")

# 11c. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [
    ("QSVM (Quantum)", y_pred_q),
    ("SVM Lineare",    yp_l),
    ("SVM RBF",        yp_r)
]
for ax, (name, preds) in zip(axes, models):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legittimo", "Frode"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12, fontweight="bold")
fig.suptitle("Confusion Matrices — Test Set (50 campioni)", fontsize=13)
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150); plt.close()
print("[PLOT] Salvato confusion_matrices.png")

# 11d. Decision boundaries (PC1 vs PC2)
def plot_decision_boundary(clf, X, y, kernel_name, ax, kernel_matrix=None):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if kernel_matrix is not None:
        # Per QSVM: ricrea il kernel tra la griglia e il train set
        # NOTA: in produzione si chiama qkernel.evaluate(grid_2d, X_train_2d)
        # Qui usiamo solo PC1, PC2 con approx RBF per visualizzazione
        from sklearn.metrics.pairwise import rbf_kernel
        K_grid = rbf_kernel(grid, X[:, :2])
        Z = clf.predict(K_grid)
    else:
        Z = clf.predict(grid)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X[y==0, 0], X[y==0, 1], c="#636EFA", s=25,
               edgecolors="white", lw=0.5, label="Legittimo", zorder=3)
    ax.scatter(X[y==1, 0], X[y==1, 1], c="#EF553B", s=35,
               marker="X", edgecolors="white", lw=0.5, label="Frode", zorder=3)
    ax.set_xlabel("PC1 [rad]"); ax.set_ylabel("PC2 [rad]")
    ax.set_title(kernel_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

# Subset 2D per visualizzazione decision boundaries
X_2d_train = X_train[:, :2]
X_2d_test  = X_test[:, :2]

svm_lin_2d = SVC(kernel="linear", C=1.0, random_state=42)
svm_lin_2d.fit(X_2d_train, y_train)

svm_rbf_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_rbf_2d.fit(X_2d_train, y_train)

# Per QSVM approx su 2D (per visualizzazione)
qsvm_2d = SVC(kernel="precomputed", C=1.0, random_state=42)
from sklearn.metrics.pairwise import rbf_kernel
K_2d = rbf_kernel(X_2d_train)
qsvm_2d.fit(K_2d, y_train)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_decision_boundary(svm_lin_2d, X_2d_train, y_train, "SVM Lineare", axes[0])
plot_decision_boundary(svm_rbf_2d, X_2d_train, y_train, "SVM RBF", axes[1])
plot_decision_boundary(qsvm_2d, X_2d_train, y_train,
                       "QSVM (approx 2D)", axes[2], kernel_matrix=True)
fig.suptitle("Decision Boundaries — PC1 vs PC2 (Training Set)",
             fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("decision_boundaries.png", dpi=150); plt.close()
print("[PLOT] Salvato decision_boundaries.png")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 12 ── SALVATAGGIO RISULTATI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 12 — SALVATAGGIO RISULTATI")
print("=" * 70)

df_res.to_csv("benchmark_results.csv")
print("[SAVE] benchmark_results.csv")
np.save("K_train.npy", K_train)
np.save("K_test.npy",  K_test)
print("[SAVE] K_train.npy, K_test.npy")
np.save("X_train.npy", X_train); np.save("y_train.npy", y_train)
np.save("X_test.npy",  X_test);  np.save("y_test.npy",  y_test)
print("[SAVE] X_train.npy, X_test.npy, y_train.npy, y_test.npy")

print("\n" + "═" * 70)
print("PIPELINE COMPLETATA CORRETTAMENTE")
print("═" * 70)
print(f"""
OUTPUT GENERATI:
  mi_scores.png          — Mutual Information feature ranking
  pca_scree.png          — Varianza spiegata PCA (4 PC)
  scatter_pc1_pc2.png    — Distribuzione 2D campioni bilanciati
  kernel_matrix.png      — Heatmap matrice kernel quantistica
  benchmark_metrics.png  — Confronto Accuracy / F1 / ROC-AUC
  confusion_matrices.png — Confusion matrix per i 3 modelli
  decision_boundaries.png— Decision boundaries 2D (PC1 vs PC2)
  benchmark_results.csv  — Tabella numerica comparativa
  K_train.npy / K_test.npy — Matrici kernel salvate
""")
