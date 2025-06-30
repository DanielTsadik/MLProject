# File: PCA/pca_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Load and Prepare Data ─────────────────────────────────────────────
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Features", "features.csv"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Results", "PCA"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["file", "label"])
y = df["label"]

# ─── Scale the Features ────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Perform PCA ───────────────────────────────────────────────────────
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# ─── Plot Explained Variance ───────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explained_variance.png"))
plt.close()

# ─── Scatter Plot of First Two Components ──────────────────────────────
plt.figure(figsize=(8, 6))
colors = ['royalblue', 'crimson']
labels = sorted(y.unique())
for label, color in zip(labels, colors):
    idx = (y == label)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, label=label, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter Plot (First Two Components)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter_plot.png"))
plt.close()

# ─── Top Contributing Features to PC1 ──────────────────────────────────
pc1 = pca.components_[0]
feature_contributions = pd.Series(pc1, index=X.columns).abs()
top_features = feature_contributions.sort_values(ascending=False).head(10)

# Save plot
plt.figure(figsize=(10, 5))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 10 Features Contributing to PC1")
plt.xlabel("Absolute Weight")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pc1_top_features.png"))
plt.close()

# Save CSV of top features
top_features.to_csv(os.path.join(OUTPUT_DIR, "top_pc1_features.csv"))

# Save explained variance values
pd.DataFrame({
    "Component": np.arange(1, len(explained_variance) + 1),
    "Cumulative Explained Variance": explained_variance
}).to_csv(os.path.join(OUTPUT_DIR, "explained_variance.txt"), index=False)