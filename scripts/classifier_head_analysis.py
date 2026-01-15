from pathlib import Path
import sys
import os
import time
import torch
import numpy as np
from metrics import plot_layer_performance

from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut

# --- Paths / Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOMIC_FM_DIR = REPO_ROOT / "external" / "genomic-FM"
DATA_DIR = GENOMIC_FM_DIR / "root" / "data"  # where verified_real_clinvar.csv likely lives

print("\n[1/6] Setting up import paths...")
print(f"  - REPO_ROOT:     {REPO_ROOT}")
print(f"  - GENOMIC_FM_DIR:{GENOMIC_FM_DIR}")
print(f"  - DATA_DIR:      {DATA_DIR}")

sys.path.insert(0, str(GENOMIC_FM_DIR))
os.chdir(GENOMIC_FM_DIR)

# Load Embeddings
path = "root/data/clinvar_pooled_embeddings__n155__bp3000__tok505__layers10.pt"
payload = torch.load(path, map_location="cpu")

print("Loading embeddings from:", path)

layers = payload["layers"]      # store all layer information
labels = np.array(payload["labels"])        # store labels
n = len(labels)  # number of variants 

# Construct Logistic Regression Classifier Pipeline
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("PCA", PCA(n_components=50)),  # optional: reduce dimensionality
    ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

# Analyze Each Layer and Conduct Stratified K-Fold Cross-Validation
print("\nPerforming Stratified K-Fold Cross-Validation for each layer on DELTA EMBEDDINGS...")
layers = [1, 3, 5, 9, 12, 15, 18, 22, 25, 28]  # 0..28 indexing
aucs = []
pres = []
recs = []

for layer in layers:
    E = payload["embeddings_by_layer"][layer]  # shape (2N, 1024), torch.Tensor

    ref = E[:n].numpy()      # (N, 1024)s
    alt = E[n:].numpy()      # (N, 1024)
    X = alt - ref            # (N, 1024)    Compute deltas

    # Binary classification: Class 1 vs Class 5
    mask = np.isin(labels, ["Class 1", "Class 2", "Class 4", "Class 5"])
    X_bin = X[mask]
    y_bin = (labels[mask] == ("Class 5" or "Class 4")).astype(int)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scoring = ['roc_auc', 'precision', 'recall']
    cv_results = cross_validate(clf, X_bin, y_bin, cv=kfolds, scoring=scoring)

    auc_mean = cv_results['test_roc_auc'].mean()
    auc_std  = cv_results['test_roc_auc'].std()
    aucs.append([layer, auc_mean, auc_std])

    pre_mean = cv_results['test_precision'].mean()
    pre_std  = cv_results['test_precision'].std()
    pres.append([layer, pre_mean, pre_std])

    rec_mean = cv_results['test_recall'].mean()
    rec_std  = cv_results['test_recall'].std()
    recs.append([layer, rec_mean, rec_std])

    print(f"Layer {layer:>2}: "
        f"AUC {auc_mean:.3f} ± {auc_std:.3f} | "
        f"PRE {pre_mean:.3f} ± {pre_std:.3f} | "
        f"REC {rec_mean:.3f} ± {rec_std:.3f}")

plot_layer_performance(aucs, pres, recs)



# REF ANALYSIS
"""
print("\nPerforming Stratified K-Fold Cross-Validation for each layer on REF EMBEDDINGS...")
layers = [1, 3, 5, 9, 12, 15, 18, 22, 25, 28]  # 0..28 indexing
for layer in layers:
    E = payload["embeddings_by_layer"][layer]  # shape (2N, 1024), torch.Tensor

    X = E[:n].numpy()      # (N, 1024)

    # Binary classification: Class 1 vs Class 5
    mask = np.isin(labels, ["Class 1", "Class 5"])
    X_bin = X[mask]
    y_bin = (labels[mask] == "Class 5").astype(int)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    auc = cross_val_score(clf, X_bin, y_bin, cv=kfolds, scoring="roc_auc")
    print(f"Layer {layer:>2}: ROC-AUC {auc.mean():.3f} +/- {auc.std():.3f}")


# ALT ANALYSIS
print("\nPerforming Stratified K-Fold Cross-Validation for each layer on ALT EMBEDDINGS...")
layers = [1, 3, 5, 9, 12, 15, 18, 22, 25, 28]  # 0..28 indexing
for layer in layers:
    E = payload["embeddings_by_layer"][layer]  # shape (2N, 1024), torch.Tensor

    X = E[n:].numpy()      # (N, 1024)

    # Binary classification: Class 1 vs Class 5
    mask = np.isin(labels, ["Class 1", "Class 5"])
    X_bin = X[mask]
    y_bin = (labels[mask] == "Class 5").astype(int)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    auc = cross_val_score(clf, X_bin, y_bin, cv=kfolds, scoring="roc_auc")
    print(f"Layer {layer:>2}: ROC-AUC {auc.mean():.3f} +/- {auc.std():.3f}")
"""