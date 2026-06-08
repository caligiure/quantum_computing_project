#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# QSVM COMPLETA v3 — Bank Account Fraud (BAF NeurIPS 2022)
# Miglioramenti v3:
#   1. Campionamento stratificato intelligente (K-Means boundary sampling)
#   2. PCA (3 componenti) + top-1 feature MI → 4 dimensioni più discriminanti
#   3. Grid Search su C per QSVM e SVM classiche
#   4. N_SHOTS aumentato a 16384
# Requisiti: pip install qiskit qiskit-aer scikit-learn pandas numpy
#             matplotlib seaborn imbalanced-learn
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 1 ── CARICAMENTO E ISPEZIONE DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("BLOCCO 1 — CARICAMENTO DATASET")
print("=" * 70)

df = pd.read_csv("Base.csv")

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

# 2b. Encoding automatico di tutte le colonne stringa/categoriche
CAT = df.select_dtypes(include=["object", "category"]).columns.tolist()
CAT = [c for c in CAT if c != "fraud_bool"]
for col in CAT:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print(f"[ENCODE] LabelEncoded ({len(CAT)} colonne): {CAT}")

# 2b-bis. Controllo di sicurezza: forza conversione numerica su tutto
for col in df.columns:
    if col == "fraud_bool":
        continue
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [ENCODE extra] {col} codificata in secondo passaggio")

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

# 2e. StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(f"[SCALE] StandardScaler applicato su {X.shape[1]} feature")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 3 ── FEATURE SELECTION: MUTUAL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 3 — FEATURE SELECTION: MUTUAL INFORMATION")
print("=" * 70)

mi = mutual_info_classif(X_scaled, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("Top 10 feature per Mutual Information (I(X;Y)):")
print(mi_series.head(10).to_string())

# Seleziona la top-1 feature MI: sarà la 4a dimensione (oltre le 3 PC)
top1_mi_feature = mi_series.index[0]
print(f"\n[MI] Top-1 feature selezionata per la 4a dimensione: '{top1_mi_feature}'")

# Plot MI scores
fig, ax = plt.subplots(figsize=(10, 5))
mi_series.head(15).plot(kind="barh", ax=ax, color="#01696f")
ax.axvline(mi_series[top1_mi_feature], color="#EF553B", lw=2,
           linestyle="--", label=f"Soglia top-1: {top1_mi_feature}")
ax.set_xlabel("Mutual Information Score", fontsize=12)
ax.set_title("Feature Importance — Mutual Information vs fraud_bool", fontsize=13)
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.savefig("mi_scores.png", dpi=150)
plt.close()
print("[PLOT] Salvato mi_scores.png")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 4 ── FEATURE PROJECTION: PCA (3 PC) + TOP-1 MI FEATURE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 4 — FEATURE PROJECTION: PCA (3 PC) + TOP-1 MI FEATURE")
print("=" * 70)

N_QUBITS = 4
N_PCA    = 3   # 3 componenti PCA + 1 feature MI = 4 dimensioni totali

# 4a. PCA su 3 componenti
pca = PCA(n_components=N_PCA, random_state=42)
X_pca3 = pca.fit_transform(X_scaled)

ev_ratio   = pca.explained_variance_ratio_
cumulative = np.cumsum(ev_ratio)
print("Varianza spiegata (3 PC):")
for i, (ev, cum) in enumerate(zip(ev_ratio, cumulative)):
    print(f"  PC{i+1}: {ev*100:.2f}%  (cumulativo: {cum*100:.2f}%)")

# 4b. Aggiunta della top-1 MI feature come 4a dimensione
# Già standardizzata in X_scaled_df → pronta all'uso
top1_scaled = X_scaled_df[top1_mi_feature].values.reshape(-1, 1)
X_combined  = np.hstack([X_pca3, top1_scaled])   # shape: (N, 4)
print(f"\n[COMBINE] Shape finale: {X_combined.shape}")
print(f"  Dimensioni: PC1, PC2, PC3, {top1_mi_feature} (standardizzata)")

# Plot scree
fig, ax = plt.subplots(figsize=(7, 4))
labels = [f"PC{i+1}" for i in range(N_PCA)] + [f"MI:{top1_mi_feature[:8]}"]
variances = list(ev_ratio * 100) + [mi_series[top1_mi_feature] * 100]
ax.bar(labels, variances, color=["#01696f"]*N_PCA + ["#EF553B"], alpha=0.85)
ax.plot(labels[:N_PCA], cumulative * 100, "o--", color="#636EFA",
        lw=2, label="Cumulativo PCA")
for i, c in enumerate(cumulative):
    ax.annotate(f"{c*100:.1f}%", (i, c*100 + 0.5), ha="center", fontsize=9)
ax.set_ylabel("Varianza / MI Score (%)")
ax.set_title("Proiezione Feature — 3 PC + 1 MI Feature")
ax.legend()
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150)
plt.close()
print("[PLOT] Salvato pca_scree.png")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 5 ── CAMPIONAMENTO STRATIFICATO INTELLIGENTE (K-Means Boundary)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 5 — CAMPIONAMENTO STRATIFICATO INTELLIGENTE (K-Means)")
print("=" * 70)

N_CLASS = 150   # campioni per classe

X_combined_df = pd.DataFrame(X_combined,
                              columns=[f"PC{i+1}" for i in range(N_PCA)] +
                                      [top1_mi_feature])
X_combined_df["fraud_bool"] = y.values

df_legit = X_combined_df[X_combined_df["fraud_bool"] == 0]
df_fraud  = X_combined_df[X_combined_df["fraud_bool"] == 1]

feature_cols = [c for c in X_combined_df.columns if c != "fraud_bool"]

def kmeans_boundary_sample(df_source, df_opposite, n_samples, feature_cols, random_state=42):
    """
    Campiona i punti di df_source più vicini al centroide di df_opposite.
    Questi sono i campioni di 'confine' — i più informativi per la SVM.
    Fallback a resample casuale se n_samples > len(df_source).
    """
    if len(df_source) <= n_samples:
        return resample(df_source, n_samples=n_samples,
                        replace=True, random_state=random_state)

    # Centroide della classe opposta
    centroid_opp = df_opposite[feature_cols].values.mean(axis=0)

    # Distanza euclidea di ogni campione dal centroide opposto
    dists = np.linalg.norm(
        df_source[feature_cols].values - centroid_opp, axis=1
    )
    # Prendi i più vicini (= i campioni di confine, più difficili da classificare)
    idx_sorted = np.argsort(dists)
    # Mescola i top 2*n_samples per evitare determinismo eccessivo
    top_idx = idx_sorted[:min(2 * n_samples, len(df_source))]
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(top_idx, size=n_samples, replace=False)
    return df_source.iloc[chosen]

legit_sampled = kmeans_boundary_sample(df_legit, df_fraud,  N_CLASS, feature_cols)
fraud_sampled = kmeans_boundary_sample(df_fraud, df_legit,  N_CLASS, feature_cols)

bal = pd.concat([legit_sampled, fraud_sampled]).sample(frac=1, random_state=42)
print(f"Dataset bilanciato: {bal.shape}")
print(f"Classi: {bal['fraud_bool'].value_counts().to_dict()}")
print(f"Strategia: boundary sampling (campioni di confine tra le classi)")

X_bal = bal[feature_cols].values
y_bal = bal["fraud_bool"].values

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 6 ── SCALING ANGOLARE IN [-π, π]
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
ax.set_title(f"PC1 vs PC2 — {len(X_angle)} campioni (boundary sampling)")
ax.legend(); plt.tight_layout()
plt.savefig("scatter_pc1_pc2.png", dpi=150)
plt.close()
print("[PLOT] Salvato scatter_pc1_pc2.png")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 8 ── QUANTUM KERNEL (compute-uncompute manuale)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BLOCCO 8 — QUANTUM KERNEL: ZZFeatureMap (reps=1, full)")
print("=" * 70)

from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

# 8a. ZZFeatureMap: reps=1, entanglement=full
feature_map = ZZFeatureMap(
    feature_dimension=N_QUBITS,
    reps=1,
    entanglement="full"
)
print("ZZFeatureMap creata:")
print(feature_map.decompose())

backend  = AerSimulator(method="statevector")
N_SHOTS  = 16384    # aumentato da 8192 → riduce errore statistico da ~1.1% a ~0.8%

def compute_kernel_entry(xi, xj, feature_map, backend, shots=N_SHOTS):
    params  = feature_map.parameters
    circ_xi     = feature_map.assign_parameters(dict(zip(params, xi)))
    circ_xj_inv = feature_map.assign_parameters(dict(zip(params, xj))).inverse()
    n  = feature_map.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(circ_xi,     inplace=True)
    qc.compose(circ_xj_inv, inplace=True)
    qc.measure(range(n), range(n))
    qc_t   = transpile(qc, backend, optimization_level=0)
    counts = backend.run(qc_t, shots=shots).result().get_counts()
    return counts.get("0" * n, 0) / shots

def compute_kernel_matrix(X_a, X_b, feature_map, backend,
                          shots=N_SHOTS, symmetric=False):
    n_a, n_b = len(X_a), len(X_b)
    K = np.zeros((n_a, n_b))
    if symmetric:
        pairs = [(i, j) for i in range(n_a) for j in range(i, n_b)]
        total = len(pairs)
        for idx, (i, j) in enumerate(pairs):
            if idx % 100 == 0:
                print(f"  Progresso: {idx}/{total} ({idx/total*100:.1f}%)",
                      end="\r", flush=True)
            val    = compute_kernel_entry(X_a[i], X_b[j], feature_map, backend, shots)
            K[i,j] = val; K[j,i] = val
        print(f"  Completato: {total}/{total} (100.0%)          ")
    else:
        total = n_a * n_b
        for idx, (i, j) in enumerate(itertools.product(range(n_a), range(n_b))):
            if idx % 100 == 0:
                print(f"  Progresso: {idx}/{total} ({idx/total*100:.1f}%)",
                      end="\r", flush=True)
            K[i,j] = compute_kernel_entry(X_a[i], X_b[j], feature_map, backend, shots)
        print(f"  Completato: {total}/{total} (100.0%)          ")
    return K

n_train    = X_train.shape[0]
n_pairs    = n_train * (n_train + 1) // 2
print(f"\nCalcolo K_train ({n_train}x{n_train}) — {n_pairs} coppie uniche...")
print(f"  Shots: {N_SHOTS} | Stima: ~{n_pairs*0.01/60:.0f}-{n_pairs*0.025/60:.0f} min")
K_train = compute_kernel_matrix(X_train, X_train, feature_map, backend, symmetric=True)
print(f"  Diagonale media  : {np.diag(K_train).mean():.4f} (atteso: ~1.0)")
print(f"  Simmetria max err: {np.max(np.abs(K_train - K_train.T)):.6f}")
print(f"  Range valori     : [{K_train.min():.4f}, {K_train.max():.4f}]")

print(f"\nCalcolo K_test ({X_test.shape[0]}x{n_train})...")
K_test = compute_kernel_matrix(X_test, X_train, feature_map, backend, symmetric=False)
print(f"  K_test shape: {K_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 9 ── QSVM CON GRID SEARCH SU C
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 9 — QSVM + GRID SEARCH su C (kernel='precomputed')")
print("=" * 70)

C_GRID = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Grid Search su C per QSVM (5-fold CV, metrica: F1):")
best_C_q, best_f1_q = 1.0, -1.0
for c in C_GRID:
    svc_cv = SVC(kernel="precomputed", C=c, random_state=42)
    scores = cross_val_score(svc_cv, K_train, y_train, cv=cv,
                             scoring="f1", error_score=0.0)
    mean_f1 = scores.mean()
    print(f"  C={c:5.1f} → F1_cv = {mean_f1:.4f} (±{scores.std():.4f})")
    if mean_f1 > best_f1_q:
        best_f1_q, best_C_q = mean_f1, c

print(f"\n  → Miglior C QSVM: {best_C_q}  (F1_cv={best_f1_q:.4f})")

qsvm = SVC(kernel="precomputed", C=best_C_q, probability=True, random_state=42)
qsvm.fit(K_train, y_train)
print(f"  Support vectors: {qsvm.n_support_}")

y_pred_q = qsvm.predict(K_test)
y_prob_q = qsvm.predict_proba(K_test)[:, 1]

acc_q  = accuracy_score(y_test, y_pred_q)
prec_q = precision_score(y_test, y_pred_q, zero_division=0)
rec_q  = recall_score(y_test, y_pred_q, zero_division=0)
f1_q   = f1_score(y_test, y_pred_q, zero_division=0)
auc_q  = roc_auc_score(y_test, y_prob_q)
print(f"\n  [QSVM] Acc={acc_q:.4f} | Prec={prec_q:.4f} | "
      f"Rec={rec_q:.4f} | F1={f1_q:.4f} | AUC={auc_q:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 10 ── BENCHMARKING: SVM LINEARE + RBF CON GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 10 — BENCHMARKING: SVM LINEARE + RBF (Grid Search su C)")
print("=" * 70)

results = {}
results["QSVM (ZZ-Feature Map)"] = {
    "Accuracy": acc_q, "Precision": prec_q,
    "Recall": rec_q, "F1": f1_q, "ROC-AUC": auc_q,
    "Best_C": best_C_q
}

for kernel_name, kernel_type, extra in [
        ("SVM Lineare", "linear",  {}),
        ("SVM RBF",     "rbf",     {"gamma": "scale"})]:
    best_C_k, best_f1_k = 1.0, -1.0
    print(f"\nGrid Search {kernel_name}:")
    for c in C_GRID:
        svc_cv = SVC(kernel=kernel_type, C=c, **extra, random_state=42)
        scores = cross_val_score(svc_cv, X_train, y_train, cv=cv,
                                 scoring="f1", error_score=0.0)
        mean_f1 = scores.mean()
        print(f"  C={c:5.1f} → F1_cv = {mean_f1:.4f}")
        if mean_f1 > best_f1_k:
            best_f1_k, best_C_k = mean_f1, c

    print(f"  → Miglior C {kernel_name}: {best_C_k}")
    svm_best = SVC(kernel=kernel_type, C=best_C_k, **extra,
                   probability=True, random_state=42)
    svm_best.fit(X_train, y_train)
    yp  = svm_best.predict(X_test)
    ypp = svm_best.predict_proba(X_test)[:, 1]
    results[kernel_name] = {
        "Accuracy":  accuracy_score(y_test, yp),
        "Precision": precision_score(y_test, yp, zero_division=0),
        "Recall":    recall_score(y_test, yp, zero_division=0),
        "F1":        f1_score(y_test, yp, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, ypp),
        "Best_C":    best_C_k
    }
    if kernel_name == "SVM Lineare":
        yp_l, ypp_l = yp, ypp
    else:
        yp_r, ypp_r = yp, ypp

df_res = pd.DataFrame(results).T
print("\n── TABELLA COMPARATIVA ──────────────────────────────────────────")
print(df_res.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 11 ── VISUALIZZAZIONI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCCO 11 — VISUALIZZAZIONI")
print("=" * 70)

# 11a. Kernel matrix heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(K_train, cmap="YlGnBu", ax=ax,
            cbar_kws={"label": "Fidelity K(xi,xj)"},
            xticklabels=False, yticklabels=False)
ax.set_title(f"Matrice Kernel Quantistica (ZZFeatureMap, {N_QUBITS} qubit)\n"
             f"Training set ({X_train.shape[0]} campioni) — reps=1, full")
plt.tight_layout(); plt.savefig("kernel_matrix.png", dpi=150); plt.close()
print("[PLOT] Salvato kernel_matrix.png")

# 11b. Benchmark metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric in zip(axes, ["Accuracy", "F1", "ROC-AUC"]):
    vals  = df_res[metric]
    bars  = ax.bar(vals.index, vals.values,
                   color=["#01696f", "#636EFA", "#EF553B"], width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                f"{v:.3f}", ha="center", fontsize=10)
fig.suptitle("Confronto QSVM vs SVM Classiche — Dataset BAF NeurIPS 2022\n"
             "v3: Boundary Sampling + PCA+MI + GridSearch C + 16384 shots",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("benchmark_metrics.png", dpi=150); plt.close()
print("[PLOT] Salvato benchmark_metrics.png")

# 11c. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, preds) in zip(axes, [
        ("QSVM (Quantum)", y_pred_q),
        ("SVM Lineare",    yp_l),
        ("SVM RBF",        yp_r)]):
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legittimo", "Frode"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12, fontweight="bold")
fig.suptitle(f"Confusion Matrices — Test Set ({len(y_test)} campioni)", fontsize=13)
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150); plt.close()
print("[PLOT] Salvato confusion_matrices.png")

# 11d. Decision boundaries 2D (PC1 vs PC2)
def plot_decision_boundary(clf, X, y, title, ax, use_rbf_approx=False):
    h = 0.05
    x0_min, x0_max = X[:,0].min()-0.3, X[:,0].max()+0.3
    x1_min, x1_max = X[:,1].min()-0.3, X[:,1].max()+0.3
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h),
                         np.arange(x1_min, x1_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if use_rbf_approx:
        from sklearn.metrics.pairwise import rbf_kernel
        K_grid = rbf_kernel(grid, X[:, :2])
        Z = clf.predict(K_grid)
    else:
        Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X[y==0,0], X[y==0,1], c="#636EFA", s=25, edgecolors="white",
               lw=0.5, label="Legittimo", zorder=3)
    ax.scatter(X[y==1,0], X[y==1,1], c="#EF553B", s=35, marker="X",
               edgecolors="white", lw=0.5, label="Frode", zorder=3)
    ax.set_xlabel("PC1 [rad]"); ax.set_ylabel("PC2 [rad]")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

X_2d = X_train[:, :2]
svm_lin_2d = SVC(kernel="linear", C=results["SVM Lineare"]["Best_C"]).fit(X_2d, y_train)
svm_rbf_2d = SVC(kernel="rbf",    C=results["SVM RBF"]["Best_C"],
                 gamma="scale").fit(X_2d, y_train)
from sklearn.metrics.pairwise import rbf_kernel
K_2d   = rbf_kernel(X_2d)
qsvm_2d = SVC(kernel="precomputed",
              C=results["QSVM (ZZ-Feature Map)"]["Best_C"]).fit(K_2d, y_train)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_decision_boundary(svm_lin_2d, X_2d, y_train, "SVM Lineare", axes[0])
plot_decision_boundary(svm_rbf_2d, X_2d, y_train, "SVM RBF",     axes[1])
plot_decision_boundary(qsvm_2d,    X_2d, y_train, "QSVM (approx 2D)",
                       axes[2], use_rbf_approx=True)
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
np.save("X_train.npy", X_train); np.save("y_train.npy", y_train)
np.save("X_test.npy",  X_test);  np.save("y_test.npy",  y_test)
print("[SAVE] array numpy salvati")

print("\n" + "═" * 70)
print("PIPELINE v3 COMPLETATA")
print("═" * 70)
print("""
OUTPUT:
  mi_scores.png           — MI feature ranking + soglia top-1
  pca_scree.png           — 3 PC + MI feature (varianza)
  scatter_pc1_pc2.png     — distribuzione boundary-sampled
  kernel_matrix.png       — heatmap K_train
  benchmark_metrics.png   — Accuracy / F1 / ROC-AUC (v3)
  confusion_matrices.png  — confusion matrix 3 modelli
  decision_boundaries.png — decision boundaries 2D
  benchmark_results.csv   — tabella numerica con Best_C
""")
