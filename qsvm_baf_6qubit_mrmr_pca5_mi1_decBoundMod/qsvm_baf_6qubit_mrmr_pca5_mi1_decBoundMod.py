#!/usr/bin/env python3
# =============================================================================
# QSVM 6 QUBIT — BAF NeurIPS 2022
# Pipeline: preprocessing -> mRMR + top-MI feature -> PCA(5) + top-MI feature -> ZZFeatureMap(6 qubit)
# Obiettivo:
# 1. Aumentare la varianza spiegata usando 5 componenti PCA invece di 4
# 2. Conservare esplicitamente una feature supervisionata molto informativa
#    sul 6° qubit tramite Mutual Information
# 3. Introdurre una selezione supervisionata mRMR per tracciare le feature più
#    rilevanti e meno ridondanti rispetto al target
# 4. Salvare anche il rendering del circuito ZZFeatureMap in formato PDF e PNG
# Requisiti:
# pip install qiskit qiskit-aer scikit-learn pandas numpy matplotlib seaborn imbalanced-learn mrmr-selection pylatexenc
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
from mrmr import mrmr_classif

print("=" * 84)
print("QSVM 6 QUBIT — PCA(5) + TOP FEATURE MI")
print("=" * 84)
print("Questo script estende la pipeline 5-qubit aggiungendo:")
print(" - PC5 come quinta componente principale")
print(" - una top feature supervisionata da Mutual Information come 6° input")
print(" - una ZZFeatureMap con feature_dimension = 6")
print(" - un blocco mRMR supervisionato per tracciare feature rilevanti e non ridondanti")

# -----------------------------------------------------------------------------
# BLOCCO 1 — Caricamento dataset
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 1 — CARICAMENTO DATASET")
print("=" * 70)

df = pd.read_csv("Base.csv")

print(f"Shape originale: {df.shape}")
print(f"Distribuzione classi: {df['fraud_bool'].value_counts().to_dict()}")
print(f"Percentuale frodi: {df['fraud_bool'].mean()*100:.2f}%")
print(f"Missing totali codificati come -1: {(df == -1).sum().sum()}")

# -----------------------------------------------------------------------------
# BLOCCO 2 — Preprocessing
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 2 — PREPROCESSING")
print("=" * 70)

DROP = ["device_fraud_count", "month"]
df.drop(columns=DROP, inplace=True, errors="ignore")
print(f"[DROP] colonne rimosse: {DROP}")

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "fraud_bool"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print(f"[ENCODE] LabelEncoded ({len(cat_cols)} colonne): {cat_cols}")

for col in df.columns:
    if col == "fraud_bool":
        continue
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"[ENCODE extra] {col} codificata in un secondo passaggio")

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

X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]
print(f"[SPLIT] X shape = {X.shape} | y shape = {y.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(f"[SCALE] StandardScaler applicato su {X.shape[1]} feature")

# -----------------------------------------------------------------------------
# BLOCCO 3 — FEATURE SELECTION mRMR (supervisionata, minima ridondanza)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 3 — FEATURE SELECTION mRMR (Minimum Redundancy Maximum Relevance)")
print("=" * 70)

N_PCA = 5
N_QUBITS = 6
N_FEATURES_TO_SELECT = N_QUBITS - 1
print(f"[MRMR] Seleziono le top {N_FEATURES_TO_SELECT} feature non ridondanti per il target")

selected_features_mrmr = mrmr_classif(
    X=X_scaled_df,
    y=y,
    K=N_FEATURES_TO_SELECT
)

print(f"\n[MRMR] Feature selezionate ({len(selected_features_mrmr)}):")
for rank, feat in enumerate(selected_features_mrmr, 1):
    print(f"   [MRMR-{rank:02d}] {feat}")

mi_scores = mutual_info_classif(X_scaled_df, y, random_state=42)
mi_df = pd.DataFrame(mi_scores, index=X_scaled_df.columns, columns=["MI"])
mi_df_sorted = mi_df.sort_values("MI", ascending=False)
top1_mi_feature = mi_df_sorted.index[0]
top1_mi_score = mi_df_sorted.iloc[0, 0]

print("\nTop 10 feature per Mutual Information:")
print(mi_df_sorted.head(10).to_string())

# Assicuriamoci che la top MI sia diversa dalle mRMR (non duplicata)
if top1_mi_feature in selected_features_mrmr:
    # La top MI è già stata presa da mRMR → non serve aggiungerla di nuovo
    print(f"\n[MRMR] Nota: la top MI feature '{top1_mi_feature}' è già nel set mRMR")
    final_feature_list = selected_features_mrmr
else:
    # Aggiungiamo la top MI come feature in più
    final_feature_list = selected_features_mrmr + [top1_mi_feature]
    print(f"\n[MRMR] Top MI feature libera aggiunta: {top1_mi_feature} (MI={top1_mi_score:.6f})")

print(f"[MRMR] Lista finale feature ({len(final_feature_list)}): {final_feature_list}")
N_QUBITS = len(final_feature_list)

fig, ax = plt.subplots(figsize=(10, 5))
mi_df_sorted["MI"].head(15).plot(kind="barh", ax=ax, color="#01696f")
ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Importance — Mutual Information vs fraud_bool")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("mi_scores_6q.png", dpi=150)
plt.close()
print("[PLOT] Salvato mi_scores_6q.png")

# -----------------------------------------------------------------------------
# BLOCCO 4 — Feature Projection: PCA a 5 componenti + 1 feature MI
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 4 — FEATURE PROJECTION: PCA -> 5 COMPONENTI")
print("=" * 70)

# Applica PCA alle sole feature selezionate da mRMR (supervisionate)
X_mrmr = X_scaled_df[selected_features_mrmr].copy()
n_feat_mrmr = X_mrmr.shape[1]

# Numero di componenti PCA = min(5, numero di feature disponibili)
N_PCA = min(5, n_feat_mrmr)
pca = PCA(n_components=N_PCA, random_state=42)
X_pca = pca.fit_transform(X_mrmr)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"[PCA] PCA applicata su feature mRMR: {selected_features_mrmr}")
print("Varianza spiegata dalle componenti PCA:")
for i, (ev, cum) in enumerate(zip(explained, cumulative), start=1):
    print(f" PC{i}: {ev*100:.2f}% (cumulativo: {cum*100:.2f}%)")

print(f"\n[INFO] PCA({N_PCA}) spiega {cumulative[-1]*100:.2f}% della varianza sulle feature mRMR")

# Se la top MI è già tra le mRMR, non la aggiungiamo di nuovo
if top1_mi_feature in selected_features_mrmr:
    X_6 = X_pca
    feature_cols_6 = [f"PC{i}" for i in range(1, N_PCA + 1)]
    print(f"[INFO] La top MI '{top1_mi_feature}' è già inclusa nelle feature mRMR usate dalla PCA")
    print("[INFO] Nessuna feature aggiuntiva separata viene concatenata")
else:
    mi_feature_values = X_scaled_df[[top1_mi_feature]].values
    X_6 = np.hstack([X_pca, mi_feature_values])
    feature_cols_6 = [f"PC{i}" for i in range(1, N_PCA + 1)] + [top1_mi_feature]
    print(f"[INFO] Aggiunta feature MI separata: {top1_mi_feature}")

N_QUBITS = X_6.shape[1]

print(f"[INFO] Feature finali per i qubit ({N_QUBITS}): {feature_cols_6}")

fig, ax = plt.subplots(figsize=(8.5, 4.8))
labels = [f"PC{i}" for i in range(1, N_PCA + 1)]
ax.bar(labels, explained * 100, color="#01696f", alpha=0.88)
ax.plot(labels, cumulative * 100, "o--", color="#EF553B", lw=2, label="Cumulativo PCA")
for i, c in enumerate(cumulative):
    ax.annotate(f"{c*100:.1f}%", (i, c*100 + 0.6), ha="center", fontsize=10)
ax.axhline(
    cumulative[-1] * 100,
    color="#636EFA",
    linestyle=":",
    lw=2,
    label=f"PCA({N_PCA}) = {cumulative[-1]*100:.1f}%"
)
ax.set_ylabel("Varianza spiegata (%)")
ax.set_title("PCA sulle feature selezionate da mRMR")
ax.legend()
plt.tight_layout()
plt.savefig("pca_plus_mi_6q.png", dpi=150)
plt.close()
print("[PLOT] Salvato pca_plus_mi_6q.png")

# -----------------------------------------------------------------------------
# BLOCCO 5 — Bilanciamento classi
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 5 — BILANCIAMENTO CLASSI (150 per classe)")
print("=" * 70)

N_CLASS = 150
df_6 = pd.DataFrame(X_6, columns=feature_cols_6)
df_6["fraud_bool"] = y.values

legit = df_6[df_6["fraud_bool"] == 0]
fraud = df_6[df_6["fraud_bool"] == 1]
legit_down = resample(legit, n_samples=N_CLASS, random_state=42, replace=False)
fraud_bal = resample(fraud, n_samples=N_CLASS, random_state=42, replace=(len(fraud) < N_CLASS))

bal = pd.concat([legit_down, fraud_bal]).sample(frac=1, random_state=42)
X_bal = bal[feature_cols_6].values
y_bal = bal["fraud_bool"].values

print(f"Dataset bilanciato: {bal.shape}")
print(f"Classi bilanciate: {bal['fraud_bool'].value_counts().to_dict()}")
print(f"Feature finali usate: {feature_cols_6}")

# -----------------------------------------------------------------------------
# BLOCCO 6 — Scaling angolare [-pi, pi] per ZZFeatureMap
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
print("\n" + "=" * 70)
print("BLOCCO 7 — TRAIN/TEST SPLIT (75/25 stratificato)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_angle, y_bal, test_size=0.25, random_state=42, stratify=y_bal
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Distribuzione y_train: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Distribuzione y_test: {pd.Series(y_test).value_counts().to_dict()}")

# -----------------------------------------------------------------------------
# BLOCCO 8 — Quantum Kernel con Qiskit
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 8 — QUANTUM KERNEL: ZZFeatureMap + Fidelity-like compute-uncompute")
print("=" * 70)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
import itertools

feature_map = ZZFeatureMap(
    feature_dimension=N_QUBITS,
    reps=1,
    entanglement="full"
)

print("ZZFeatureMap 6-qubit creata.")
feature_map_txt = feature_map.decompose().draw(output="text", fold=-1)
feature_map_txt = str(feature_map_txt)
print(feature_map_txt)
with open("zzfeaturemap_6q.txt", "w", encoding="utf-8") as f:
    f.write(feature_map_txt)
print("[SAVE] Salvato zzfeaturemap_6q.txt")

try:
    circuit_drawer(
        feature_map.decompose(),
        output="latex",
        filename="zzfeaturemap_7q.pdf",
        fold=20
    )
    print("[SAVE] Salvato zzfeaturemap_7q.pdf (rendering LaTeX)")
except Exception as e:
    print(f"[WARN] Impossibile salvare zzfeaturemap_7q.pdf con output='latex': {e}")

try:
    fig = circuit_drawer(
        feature_map.decompose(),
        output="mpl",
        fold=20,
        scale=0.8
    )
    fig.savefig("zzfeaturemap_6q.png", dpi=300, bbox_inches="tight")
    print("[SAVE] Salvato zzfeaturemap_6q.png (rendering Matplotlib)")
except Exception as e:
    print(f"[WARN] Impossibile salvare zzfeaturemap_6q.png con output='mpl': {e}")

try:
    backend = AerSimulator(method="statevector", device="GPU")
    backend_target = "GPU"
except Exception:
    backend = AerSimulator(method="statevector")
    backend_target = "CPU"
print(f"[BACKEND] AerSimulator in uso: {backend_target}")

N_SHOTS = 8192
print(f"[CONFIG] Shots per coppia: {N_SHOTS}")

def compute_kernel_entry(xi, xj, fmap, backend, shots=N_SHOTS):
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

print(f"\nCalcolo kernel matrix train ({X_train.shape[0]}x{X_train.shape[0]})...")
print(f" N° coppie uniche: {X_train.shape[0] * (X_train.shape[0] + 1) // 2}")
print(f" Shots per coppia: {N_SHOTS}")
K_train = compute_kernel_matrix(X_train, X_train, feature_map, backend, symmetric=True)
print(f" K_train shape: {K_train.shape}")

print(f"\nCalcolo K_test ({X_test.shape[0]}x{X_train.shape[0]})...")
K_test = compute_kernel_matrix(X_test, X_train, feature_map, backend, symmetric=False)
print(f" K_test shape: {K_test.shape}")

print(f" Diagonale media K_train: {np.diag(K_train).mean():.4f}")
print(f" Errore massimo di simmetria: {np.max(np.abs(K_train - K_train.T)):.6f}")
print(f" Range K_train: [{K_train.min():.4f}, {K_train.max():.4f}]")

# -----------------------------------------------------------------------------
# BLOCCO 9 — QSVM + benchmark classico
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 9 — SVM QUANTUM (kernel='precomputed') e confronto con SVM CLASSICA")
print("=" * 70)

qsvm = SVC(kernel="precomputed", C=1.0, probability=True, random_state=42)
qsvm.fit(K_train, y_train)
y_pred_q = qsvm.predict(K_test)
y_prob_q = qsvm.predict_proba(K_test)[:, 1]
print(f"[QSVM] Support vectors per classe: {qsvm.n_support_}")

svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm_lin.fit(X_train, y_train)
yp_l = svm_lin.predict(X_test)
yp_l_prob = svm_lin.predict_proba(X_test)[:, 1]

svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm_rbf.fit(X_train, y_train)
yp_r = svm_rbf.predict(X_test)
yp_r_prob = svm_rbf.predict_proba(X_test)[:, 1]

results = {
    "QSVM (6 qubit: PCA5 + MI1)": {
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
print("\n" + "=" * 70)
print("BLOCCO 10 — GRAFICI E VISUALIZZAZIONI")
print("=" * 70)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(K_train, cmap="YlGnBu", ax=ax,
            cbar_kws={"label": "Fidelity K(xi,xj)"},
            xticklabels=False, yticklabels=False)
ax.set_title(f"Kernel quantistico — 6 qubit\nTrain set ({X_train.shape[0]} campioni)")
plt.tight_layout()
plt.savefig("kernel_matrix_6q.png", dpi=150)
plt.close()
print("[PLOT] Salvato kernel_matrix_6q.png")

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
fig.suptitle("QSVM 6 qubit vs SVM classiche — PCA(5)+MI(1)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("benchmark_metrics_6q.png", dpi=150)
plt.close()
print("[PLOT] Salvato benchmark_metrics_6q.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [
    ("QSVM 6q", y_pred_q),
    ("SVM Lineare", yp_l),
    ("SVM RBF", yp_r)
]
for ax, (name, preds) in zip(axes, models):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legittimo", "Frode"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12, fontweight="bold")
fig.suptitle("Confusion matrices — setup 6 qubit", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrices_6q.png", dpi=150)
plt.close()
print("[PLOT] Salvato confusion_matrices_6q.png")

# Decision boundaries con proiezione globale PCA(2)
# - SVM classiche: decision boundaries nel piano 2D ottenuto con PCA globale
#   calcolata su TUTTE le feature finali di X_train
# - QSVM: embedding 2D ottenuto direttamente dalla matrice kernel quantistica K_train
#   tramite Kernel PCA con kernel='precomputed'
print("[PLOT] Generazione visualizzazione definitiva: PCA globale + quantum-kernel embedding")

from sklearn.decomposition import PCA, KernelPCA

def plot_decision_boundary(clf, X, y, model_name, ax):
    """
    Disegna la frontiera decisionale di un classificatore addestrato direttamente
    su coordinate 2D.
    """
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.30, cmap="RdBu")
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1],
        c="#636EFA", s=28,
        edgecolors="white", lw=0.5,
        label="Legittimo", zorder=3
    )
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1],
        c="#EF553B", s=36, marker="X",
        edgecolors="white", lw=0.5,
        label="Frode", zorder=3
    )
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Global PCA 1 [a.u.]")
    ax.set_ylabel("Global PCA 2 [a.u.]")
    ax.legend(fontsize=9)

# -------------------------------------------------------------------------
# 1. PCA globale di plotting per i modelli classici
# -------------------------------------------------------------------------
pca_plot = PCA(n_components=2, random_state=42)
X_plot_train = pca_plot.fit_transform(X_train)
X_plot_test = pca_plot.transform(X_test)

plot_explained = pca_plot.explained_variance_ratio_
plot_cumulative = np.cumsum(plot_explained)

print(f"[INFO] PCA globale plotting train shape: {X_plot_train.shape}")
print(f"[INFO] Varianza spiegata Global PCA 1: {plot_explained[0]*100:.2f}%")
print(f"[INFO] Varianza spiegata Global PCA 2: {plot_explained[1]*100:.2f}%")
print(f"[INFO] Varianza cumulativa plotting 2D: {plot_cumulative[-1]*100:.2f}%")

svm_lin_2d = SVC(kernel="linear", C=1.0, random_state=42)
svm_lin_2d.fit(X_plot_train, y_train)
print("[FIT] svm_lin_2d addestrata su PCA globale")

svm_rbf_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_rbf_2d.fit(X_plot_train, y_train)
print("[FIT] svm_rbf_2d addestrata su PCA globale")

# -------------------------------------------------------------------------
# 2. Embedding 2D della QSVM a partire dal kernel quantistico reale
# -------------------------------------------------------------------------
kpca_q = KernelPCA(
    n_components=2,
    kernel="precomputed",
    eigen_solver="auto"
)
X_q_embed = kpca_q.fit_transform(K_train)

print(f"[INFO] Quantum kernel embedding shape: {X_q_embed.shape}")

# -------------------------------------------------------------------------
# 3. Figura finale: due boundary classiche + embedding QSVM
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_decision_boundary(
    svm_lin_2d, X_plot_train, y_train,
    "SVM Lineare — Global PCA",
    axes[0]
)

plot_decision_boundary(
    svm_rbf_2d, X_plot_train, y_train,
    "SVM RBF — Global PCA",
    axes[1]
)

axes[2].scatter(
    X_q_embed[y_train == 0, 0], X_q_embed[y_train == 0, 1],
    c="#636EFA", s=28,
    edgecolors="white", lw=0.5,
    label="Legittimo", zorder=3
)
axes[2].scatter(
    X_q_embed[y_train == 1, 0], X_q_embed[y_train == 1, 1],
    c="#EF553B", s=36, marker="X",
    edgecolors="white", lw=0.5,
    label="Frode", zorder=3
)
axes[2].set_title("QSVM — Quantum Kernel Embedding", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Kernel PC 1 [a.u.]")
axes[2].set_ylabel("Kernel PC 2 [a.u.]")
axes[2].legend(fontsize=9)

fig.suptitle(
    "Visualizzazione definitiva — Classiche su Global PCA, QSVM su Quantum Kernel Space",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("decision_boundaries_6q_definitive.png", dpi=150)
plt.close()
print("[PLOT] Salvato decision_boundaries_6q_definitive.png")

# -----------------------------------------------------------------------------
# BLOCCO 11 — Salvataggio risultati
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BLOCCO 11 — SALVATAGGIO RISULTATI")
print("=" * 70)

df_res.to_csv("benchmark_results_6q.csv")
print("[SAVE] benchmark_results_6q.csv")
np.save("K_train_6q.npy", K_train)
np.save("K_test_6q.npy", K_test)
print("[SAVE] K_train_6q.npy, K_test_6q.npy")
np.save("X_train_6q.npy", X_train)
np.save("X_test_6q.npy", X_test)
np.save("y_train_6q.npy", y_train)
np.save("y_test_6q.npy", y_test)
print("[SAVE] X_train_6q.npy, X_test_6q.npy, y_train_6q.npy, y_test_6q.npy")

with open("readme_6q.txt", "w", encoding="utf-8") as f:
    f.write(
        "QSVM 6 qubit\n"
        f"Input finale: PC1, PC2, PC3, PC4, PC5, {top1_mi_feature}\n"
        f"Top feature MI: {top1_mi_feature} (score={top1_mi_score:.6f})\n"
        f"Varianza cumulativa PCA(5): {cumulative[-1]*100:.2f}%\n"
        "Nota: il passaggio da PCA(4) a PCA(5) aumenta la varianza spiegata.\n"
        "Il 6° qubit non aumenta la varianza PCA, ma preserva informazione supervisionata utile.\n"
        "Piu qubit non garantiscono automaticamente piu precisione: il vantaggio va verificato sperimentalmente.\n"
        "Per usare GPU NVIDIA con Qiskit Aer, installa qiskit-aer-gpu e abilita device='GPU'.\n"
    )
print("[SAVE] readme_6q.txt")

print("\n" + "=" * 84)
print("PIPELINE 6 QUBIT COMPLETATA")
print("=" * 84)
print("OUTPUT GENERATI:")
print("- mi_scores_6q.png")
print("- pca_plus_mi_6q.png")
print("- zzfeaturemap_6q.txt")
print("- zzfeaturemap_6q.pdf")
print("- zzfeaturemap_6q.png")
print("- kernel_matrix_6q.png")
print("- benchmark_metrics_6q.png")
print("- confusion_matrices_6q.png")
print("- benchmark_results_6q.csv")
print("- K_train_6q.npy / K_test_6q.npy")
print("- X_train_6q.npy / X_test_6q.npy / y_train_6q.npy / y_test_6q.npy")
print("- readme_6q.txt")
print("=" * 84)
