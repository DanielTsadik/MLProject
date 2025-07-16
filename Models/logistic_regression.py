import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "Features", "features.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "LogisticRegression")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
X = df.drop(columns=["file", "label"]).values
y = df["label"].values

# ─── Train/Test Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── Train Model ───────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─── Classification Report ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, target_names=["RealArt", "AiArtData"])
print("\n=== Logistic Regression Classification Report ===")
print(report)

with open(os.path.join(RESULTS_DIR, "logreg_classification_report.txt"), "w") as f:
    f.write(report)

# ─── Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "AI"])
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logreg_confusion_matrix.png"))
plt.close()

# ─── Bar Chart of Final Evaluation ─────────────────────────────────────
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred)
}

plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color="mediumseagreen")
plt.ylim(0, 1)
plt.title("Logistic Regression Evaluation Metrics")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logreg_scores_bar.png"))
plt.close()
