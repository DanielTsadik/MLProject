import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "Features", "features.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "KNN_CV")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
X = df.drop(columns=["file", "label"]).values
y = df["label"].values

# ─── K-Fold Cross-Validation to Select Best K ──────────────────────────
max_k = 20
kf = StratifiedKFold(n_splits=5, shuffle=True)
mean_accuracies = []
mean_f1_scores = []

print("\n=== K-Fold Cross-Validation for K selection ===")
for k in range(1, max_k + 1):
    acc_scores, f1_scores = [], []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── Train Final Model ─────────────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─── Classification Report ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, target_names=["RealArt", "AiArtData"])
print("\n=== Final Classification Report ===")
print(report)

report_path = os.path.join(RESULTS_DIR, "knn_classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Best K = {best_k}\n")
    f.write(report)

# ─── Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "AI"])
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title(f"KNN Confusion Matrix (K = {best_k})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_confusion_matrix.png"))
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
plt.title(f"KNN Evaluation Metrics (K = {best_k})")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_scores_bar.png"))
plt.close()

# ─── Plot Accuracy / F1 vs K ───────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), mean_accuracies, marker='o', label="Accuracy")
plt.plot(range(1, max_k + 1), mean_f1_scores, marker='s', label="F1-score")
plt.axvline(best_k, color='gray', linestyle='--', label=f"Best K = {best_k}")
plt.title("KNN (Cross-Validation): Accuracy and F1 vs K")
plt.xlabel("K")
plt.ylabel("Score")
plt.xticks(range(1, max_k + 1))
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "knn_k_selection_plot.png"))
plt.close()