"""
Visualization script for attention pattern analysis
Run this to generate all EDA visualizations
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.figsize"] = (14, 8)

# Create output directory
os.makedirs("../results/visualizations", exist_ok=True)


def load_data():
    """Load the processed dataset"""
    print("Loading dataset...")
    df = pd.read_csv("../data/processed/attention_dataset.csv")
    print(f"✓ Loaded {len(df)} samples")
    return df


def create_all_visualizations(df):
    """Generate all visualization plots"""

    # Get feature columns
    feature_cols = [
        c for c in df.columns if c not in ["task_type", "prompt", "n_tokens"]
    ]
    entropy_cols = [c for c in feature_cols if c.startswith("entropy")]
    induction_cols = [c for c in feature_cols if c.startswith("induction")]

    # Calculate aggregate metrics
    df["mean_entropy"] = df[entropy_cols].mean(axis=1)
    df["mean_induction"] = df[induction_cols].mean(axis=1)

    # 1. Entropy by task
    print("\nGenerating entropy visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="task_type", y="mean_entropy", palette="colorblind", ax=ax)
    ax.set_xlabel("Task Type", fontweight="bold")
    ax.set_ylabel("Mean Attention Entropy", fontweight="bold")
    ax.set_title("Attention Entropy by Task Type", fontweight="bold", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "../results/visualizations/entropy_by_task.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. PCA
    print("Generating PCA visualization...")
    X = df[feature_cols].values
    y = df["task_type"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    for task in df["task_type"].unique():
        mask = y == task
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=task, alpha=0.7, s=100)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontweight="bold")
    ax.set_title("PCA: Task Separation", fontweight="bold", fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        "../results/visualizations/pca_task_separation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("✓ All visualizations generated!")


if __name__ == "__main__":
    df = load_data()
    create_all_visualizations(df)
    print("\n✓ Visualization complete! Check results/visualizations/")
