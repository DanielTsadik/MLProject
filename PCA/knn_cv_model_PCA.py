import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "Features", "features.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "KNN_CV_PCA")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
X = df.drop(columns=["file", "label"]).values
y = df["label"].values

# ─── Standardize Data ──────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Apply PCA ─────────────────────────────────────────────────────────
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

# ─── K-Fold Cross-Validation to Select Best K ──────────────────────────
max_k = 20
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_accuracies = []
mean_f1_scores = []

print("\n=== K-Fold Cross-Validation for K selection (with PCA) ===")
for k in range(1, max_k + 1):
    acc_scores, f1_scores = [], []
    for train_idx, val_idx in kf.split(X_pca, y):
        X_train, X_val = X_pca[train_idx], X_pca[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))

    mean_acc = np.mean(acc_scores)
    mean_f1 = np.mean(f1_scores)
    mean_accuracies.append(mean_acc)
    mean_f1_scores.append(mean_f1)

    print(f"K={k:2d} | Mean Accuracy: {mean_acc:.4f} | Mean F1-score: {mean_f1:.4f}")

best_k = np.argmax(mean_f1_scores) + 1
print(f"\n✅ Best K based on F1-score: K = {best_k} (F1 = {mean_f1_scores[best_k - 1]:.4f})")

# ─── Final Train/Test Split ────────────────────────────────────────────
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize & PCA on train/test split
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_full)
X_test_scaled = scaler_final.transform(X_test_full)

pca_final = PCA(n_components=30)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

# ─── Train Final Model ─────────────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

# ─── Classification Report ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, target_names=["RealArt", "AiArtData"])
print("\n=== Final Classification Report (with PCA) ===")
print(report)

report_path = os.path.join(RESULTS_DIR, "knn_classification_report_pca.txt")
with open(report_path, "w") as f:
    f.write(f"Best K = {best_k}\n")
    f.write(report)

# ─── Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "AI"])
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title(f"KNN Confusion Matrix with PCA (K = {best_k})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_confusion_matrix_pca.png"))
plt.close()

# ─── Final Bar Chart of Scores ─────────────────────────────────────────
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred)
}

plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color="skyblue")
plt.ylim(0, 1)
plt.title(f"KNN Evaluation Metrics with PCA (K = {best_k})")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_scores_bar_pca.png"))
plt.close()

# ─── Plot Accuracy / F1 vs K ───────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), mean_accuracies, marker='o', label="Accuracy")
plt.plot(range(1, max_k + 1), mean_f1_scores, marker='s', label="F1-score")
plt.axvline(best_k, color='gray', linestyle='--', label=f"Best K = {best_k}")
plt.title("KNN (Cross-Validation) with PCA: Accuracy and F1 vs K")
plt.xlabel("K")
plt.ylabel("Score")
plt.xticks(range(1, max_k + 1))
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_k_selection_plot_pca.png"))
plt.close()
