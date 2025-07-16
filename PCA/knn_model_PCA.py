import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "KNN_PCA")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
X = df.drop(columns=["file", "label"]).values
y = df["label"].values

# ─── Train/Test Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── Standardize Data ──────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── Apply PCA ─────────────────────────────────────────────────────────
pca = PCA(n_components=30)  # You can change this number!
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ─── Cross-validation over K values ─────────────────────────────────────
max_k = 20
accuracies = []
f1_scores = []

print("\n=== Cross-Validation over K values (with PCA) ===")
for k in range(1, max_k + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies.append(acc)
    f1_scores.append(f1)
    print(f"K={k:2d} | Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

# ─── Choose Best K ──────────────────────────────────────────────────────
best_k = np.argmax(f1_scores) + 1
print(f"\n✅ Best K based on F1-score: K = {best_k} (F1 = {f1_scores[best_k - 1]:.4f})")

# ─── Final Model with Best K ────────────────────────────────────────────
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_pca, y_train)
y_pred = best_knn.predict(X_test_pca)

# ─── Classification Report ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, target_names=["RealArt", "AiArtData"])
print("\n=== Final Classification Report (with PCA) ===")
print(report)

report_path = os.path.join(RESULTS_DIR, "knn_pca_classification_report.txt")
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

# ─── Bar Chart of Final Evaluation ─────────────────────────────────────
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

# ─── Accuracy / F1 vs K Plot ───────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), accuracies, marker='o', label="Accuracy")
plt.plot(range(1, max_k + 1), f1_scores, marker='s', label="F1-score")
plt.axvline(best_k, color='gray', linestyle='--', label=f"Best K = {best_k}")
plt.title("KNN with PCA: Accuracy and F1 vs K")
plt.xlabel("K")
plt.ylabel("Score")
plt.xticks(range(1, max_k + 1))
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_k_selection_plot_pca.png"))
plt.close()