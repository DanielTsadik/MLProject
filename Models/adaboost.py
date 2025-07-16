# File: Models/adaboost.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "Features", "features.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "Adaboost")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
X = df.drop(columns=["file", "label"]).values
y = df["label"].values

# ─── Train/Test Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── Model Training ────────────────────────────────────────────────────
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─── Classification Report ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, target_names=["RealArt", "AiArtData"])
print("\n=== Adaboost Classification Report ===")
print(report)

report_path = os.path.join(RESULTS_DIR, "adaboost_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

# ─── Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "AI"])
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Adaboost Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "adaboost_confusion_matrix.png"))
plt.close()

# ─── Metrics Bar Chart ─────────────────────────────────────────────────
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred)
}

plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color="lightgreen")
plt.ylim(0, 1)
plt.title("Adaboost Evaluation Metrics")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "adaboost_scores_bar.png"))
plt.close()