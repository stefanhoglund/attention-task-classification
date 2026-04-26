"""
Analyze which attention features are most important for task classification.
Provides interpretable insights into what the model learned.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def load_data():
    """Load dataset and classification results"""
    df = pd.read_csv("./data/processed/attention_dataset.csv")

    with open("./results/classification_report.json", "r") as f:
        class_results = json.load(f)

    return df, class_results


def analyze_top_features(df, class_results):
    """Detailed analysis of most important features"""

    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Get Random Forest feature importance
    rf_features = class_results["models"]["random_forest"]["top_features"]

    print("\nTop 20 Most Important Features for Task Classification:\n")
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Interpretation'}")
    print("-" * 95)

    interpretations = []

    for idx, feat_info in enumerate(rf_features[:20], 1):
        feature = feat_info["feature"]
        importance = feat_info["importance"]

        # Parse feature name
        # Format: metric_L{layer}H{head}
        # Example: entropy_L5H3
        parts = feature.split("_")
        metric_type = parts[0]
        layer_head = parts[1] if len(parts) > 1 else ""

        if "L" in layer_head and "H" in layer_head:
            layer = int(layer_head.split("L")[1].split("H")[0])
            head = int(layer_head.split("H")[1])

            # Interpret based on layer depth
            if layer < 4:
                layer_desc = "Early layer"
                layer_role = "syntax/position"
            elif layer < 8:
                layer_desc = "Middle layer"
                layer_role = "semantics"
            else:
                layer_desc = "Late layer"
                layer_role = "task-specific"

            # Interpret metric type
            metric_meanings = {
                "entropy": "Attention focus/spread",
                "induction": "Previous token copying",
                "self_attn": "Self-attention strength",
                "first_tok": "Initial token attention",
                "spread": "Attention distribution",
                "max_attn": "Peak attention weight",
            }

            metric_desc = metric_meanings.get(metric_type, metric_type)

            interpretation = f"{layer_desc} ({layer_role}) - {metric_desc}"
        else:
            interpretation = "Aggregate metric"

        print(f"{idx:<6} {feature:<35} {importance:<12.4f} {interpretation}")

        interpretations.append(
            {
                "rank": idx,
                "feature": feature,
                "importance": importance,
                "metric": metric_type,
                "layer": layer if "L" in layer_head else None,
                "head": head if "H" in layer_head else None,
                "interpretation": interpretation,
            }
        )

    return interpretations


def analyze_layer_distribution(interpretations):
    """Analyze which layers are most important"""

    print("\n" + "=" * 70)
    print("LAYER IMPORTANCE DISTRIBUTION")
    print("=" * 70)

    # Group by layer
    layer_importance = {}
    for feat in interpretations:
        if feat["layer"] is not None:
            layer = feat["layer"]
            if layer not in layer_importance:
                layer_importance[layer] = 0
            layer_importance[layer] += feat["importance"]

    # Sort by layer
    sorted_layers = sorted(layer_importance.items())

    print("\nTotal Feature Importance by Layer (sum of top 20 features):\n")
    print(f"{'Layer':<8} {'Importance':<15} {'Description':<30} {'Bar'}")
    print("-" * 75)

    max_importance = max(layer_importance.values()) if layer_importance else 1

    for layer, importance in sorted_layers:
        if layer < 4:
            desc = "Early (syntax/position)"
        elif layer < 8:
            desc = "Middle (semantics)"
        else:
            desc = "Late (task-specific)"

        bar_length = int((importance / max_importance) * 30)
        bar = "█" * bar_length

        print(f"{layer:<8} {importance:<15.4f} {desc:<30} {bar}")

    # Summary
    early_layers = sum(imp for l, imp in sorted_layers if l < 4)
    middle_layers = sum(imp for l, imp in sorted_layers if 4 <= l < 8)
    late_layers = sum(imp for l, imp in sorted_layers if l >= 8)

    print("\nSummary:")
    print(f"  Early layers (0-3):  {early_layers:.4f}")
    print(f"  Middle layers (4-7): {middle_layers:.4f}")
    print(f"  Late layers (8-11):  {late_layers:.4f}")

    if late_layers > early_layers:
        print(
            "\n✓ Late layers are more important - supports H2 (task-specific processing)"
        )
    else:
        print("\n~ Early layers dominate - suggests syntactic features matter more")

    return layer_importance


def analyze_metric_distribution(interpretations):
    """Analyze which metric types are most important"""

    print("\n" + "=" * 70)
    print("METRIC TYPE IMPORTANCE DISTRIBUTION")
    print("=" * 70)

    # Group by metric type
    metric_importance = {}
    for feat in interpretations:
        metric = feat["metric"]
        if metric not in metric_importance:
            metric_importance[metric] = 0
        metric_importance[metric] += feat["importance"]

    # Sort by importance
    sorted_metrics = sorted(metric_importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTotal Feature Importance by Metric Type:\n")
    print(f"{'Metric':<15} {'Importance':<15} {'Interpretation':<40} {'Bar'}")
    print("-" * 90)

    metric_meanings = {
        "entropy": "How focused vs. diffuse attention is",
        "induction": "Copying from previous tokens",
        "self_attn": "Self-attention (diagonal) strength",
        "first_tok": "Attention to first/BOS token",
        "spread": "Std dev of attention weights",
        "max_attn": "Peak attention value",
    }

    max_importance = max(metric_importance.values()) if metric_importance else 1

    for metric, importance in sorted_metrics:
        meaning = metric_meanings.get(metric, "Unknown")
        bar_length = int((importance / max_importance) * 25)
        bar = "█" * bar_length

        print(f"{metric:<15} {importance:<15.4f} {meaning:<40} {bar}")

    return metric_importance


def analyze_specific_heads(df, interpretations):
    """Analyze the most important specific attention heads"""

    print("\n" + "=" * 70)
    print("TOP ATTENTION HEADS ANALYSIS")
    print("=" * 70)

    # Get top 5 heads
    top_heads = []
    for feat in interpretations[:5]:
        if feat["layer"] is not None:
            top_heads.append(feat)

    print(f"\nDetailed Analysis of Top {len(top_heads)} Most Important Heads:\n")

    for feat in top_heads:
        layer = feat["layer"]
        head = feat["head"]
        metric = feat["metric"]
        feature_name = feat["feature"]

        print(f"\n{feat['rank']}. Layer {layer}, Head {head} - {metric.upper()}")
        print(f"   Feature: {feature_name}")
        print(f"   Importance: {feat['importance']:.4f}")
        print(f"   {'-' * 65}")

        # Get values for each task
        if feature_name in df.columns:
            task_stats = df.groupby("task_type")[feature_name].agg(["mean", "std"])

            print(f"   {'Task':<30} {'Mean':<12} {'Std':<12}")
            print(f"   {'-' * 55}")

            for task, row in task_stats.iterrows():
                print(f"   {task:<30} {row['mean']:<12.4f} {row['std']:<12.4f}")

            # Determine what this head is detecting
            task_means = task_stats["mean"]
            max_task = task_means.idxmax()
            min_task = task_means.idxmin()
            range_val = task_means.max() - task_means.min()

            print("\n   Insight:")
            print(f"   - Highest for: {max_task} ({task_means.max():.4f})")
            print(f"   - Lowest for:  {min_task} ({task_means.min():.4f})")
            print(f"   - Range: {range_val:.4f}")

            # Interpretation
            if metric == "entropy":
                if max_task == "creative_generation":
                    print("   → This head shows DIFFUSE attention for creative tasks")
                elif max_task == "arithmetic":
                    print("   → This head shows FOCUSED attention for arithmetic")
            elif metric == "induction":
                if max_task == "logical_inference":
                    print("   → This head copies patterns for logical tasks")
            elif metric == "first_tok":
                print("   → This head uses BOS token attention differently per task")


def visualize_feature_importance(interpretations, layer_importance, metric_importance):
    """Create visualization of feature importance"""

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top 10 features bar plot
    ax = axes[0, 0]
    top_10 = interpretations[:10]
    features = [f["feature"] for f in top_10]
    importances = [f["importance"] for f in top_10]

    colors = sns.color_palette("viridis", len(features))
    bars = ax.barh(range(len(features)), importances, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Importance", fontweight="bold")
    ax.set_title("Top 10 Most Important Features", fontweight="bold", fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # 2. Layer distribution
    ax = axes[0, 1]
    if layer_importance:
        layers = sorted(layer_importance.keys())
        layer_imps = [layer_importance[l] for l in layers]

        colors_by_depth = []
        for l in layers:
            if l < 4:
                colors_by_depth.append("lightcoral")
            elif l < 8:
                colors_by_depth.append("gold")
            else:
                colors_by_depth.append("lightgreen")

        ax.bar(
            layers, layer_imps, color=colors_by_depth, edgecolor="black", linewidth=1.5
        )
        ax.set_xlabel("Layer", fontweight="bold")
        ax.set_ylabel("Total Importance", fontweight="bold")
        ax.set_title("Feature Importance by Layer", fontweight="bold", fontsize=14)
        ax.set_xticks(range(12))
        ax.grid(axis="y", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightcoral", label="Early (0-3)"),
            Patch(facecolor="gold", label="Middle (4-7)"),
            Patch(facecolor="lightgreen", label="Late (8-11)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    # 3. Metric type distribution
    ax = axes[1, 0]
    metrics = list(metric_importance.keys())
    metric_imps = list(metric_importance.values())

    colors = sns.color_palette("Set2", len(metrics))
    ax.bar(metrics, metric_imps, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Metric Type", fontweight="bold")
    ax.set_ylabel("Total Importance", fontweight="bold")
    ax.set_title("Feature Importance by Metric Type", fontweight="bold", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # 4. Layer x Metric heatmap
    ax = axes[1, 1]

    # Create layer x metric matrix
    layer_metric_importance = {}
    for feat in interpretations:
        if feat["layer"] is not None:
            key = (feat["layer"], feat["metric"])
            if key not in layer_metric_importance:
                layer_metric_importance[key] = 0
            layer_metric_importance[key] += feat["importance"]

    # Create matrix
    all_metrics = list(metric_importance.keys())
    all_layers = sorted(
        set(feat["layer"] for feat in interpretations if feat["layer"] is not None)
    )

    matrix = np.zeros((len(all_layers), len(all_metrics)))
    for i, layer in enumerate(all_layers):
        for j, metric in enumerate(all_metrics):
            matrix[i, j] = layer_metric_importance.get((layer, metric), 0)

    sns.heatmap(
        matrix,
        xticklabels=all_metrics,
        yticklabels=all_layers,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        ax=ax,
        cbar_kws={"label": "Importance"},
    )
    ax.set_xlabel("Metric Type", fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    ax.set_title("Layer × Metric Importance Heatmap", fontweight="bold", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        "./results/visualizations/feature_importance_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Saved: results/visualizations/feature_importance_analysis.png")
    plt.show()


def generate_interpretation_summary(
    interpretations, layer_importance, metric_importance
):
    """Generate human-readable summary"""

    print("\n" + "=" * 70)
    print("INTERPRETATION SUMMARY FOR REPORT")
    print("=" * 70)

    print("\n### Key Findings:\n")

    # 1. Most important single feature
    top_feat = interpretations[0]
    print(f"1. **Most Discriminative Feature:** {top_feat['feature']}")
    print(f"   - Layer {top_feat['layer']}, Head {top_feat['head']}")
    print(f"   - Measures: {top_feat['metric']}")
    print(f"   - Importance: {top_feat['importance']:.4f}")
    print(
        f"   - This suggests the model relies heavily on {top_feat['interpretation']}"
    )

    # 2. Layer distribution
    early = sum(imp for l, imp in layer_importance.items() if l < 4)
    middle = sum(imp for l, imp in layer_importance.items() if 4 <= l < 8)
    late = sum(imp for l, imp in layer_importance.items() if l >= 8)
    total = early + middle + late

    print("\n2. **Layer Distribution:**")
    print(f"   - Early layers (0-3):  {early / total * 100:.1f}% of importance")
    print(f"   - Middle layers (4-7): {middle / total * 100:.1f}% of importance")
    print(f"   - Late layers (8-11):  {late / total * 100:.1f}% of importance")

    if late > early and late > middle:
        print("   - **Supports H2:** Late layers show most task-specific patterns")

    # 3. Metric distribution
    top_metric = max(metric_importance.items(), key=lambda x: x[1])
    print(f"\n3. **Most Important Metric Type:** {top_metric[0]}")
    print(
        f"   - Contributes {top_metric[1] / sum(metric_importance.values()) * 100:.1f}% of total importance"
    )

    metric_meanings = {
        "entropy": "Attention focus is the primary task differentiator",
        "induction": "Pattern copying behavior distinguishes tasks",
        "spread": "Attention distribution variance is key",
        "self_attn": "Self-attention strength differentiates tasks",
    }

    if top_metric[0] in metric_meanings:
        print(f"   - Interpretation: {metric_meanings[top_metric[0]]}")

    print("\n4. **Practical Implications:**")
    print("   - The Random Forest model identified specific attention heads")
    print("   - These heads show clear task-specific behavior patterns")
    print("   - Can be used for model debugging and interpretability")
    print("   - Suggests transformers develop specialized circuits for different tasks")


def main():
    """Run complete feature importance analysis"""

    # Load data
    df, class_results = load_data()

    # Analyze top features
    interpretations = analyze_top_features(df, class_results)

    # Analyze layer distribution
    layer_importance = analyze_layer_distribution(interpretations)

    # Analyze metric distribution
    metric_importance = analyze_metric_distribution(interpretations)

    # Analyze specific heads
    analyze_specific_heads(df, interpretations)

    # Visualize
    visualize_feature_importance(interpretations, layer_importance, metric_importance)

    # Generate summary
    generate_interpretation_summary(
        interpretations, layer_importance, metric_importance
    )

    # Save detailed analysis
    analysis_output = {
        "top_features": interpretations,
        "layer_importance": {int(k): float(v) for k, v in layer_importance.items()},
        "metric_importance": {k: float(v) for k, v in metric_importance.items()},
    }

    with open("./results/feature_importance_detailed.json", "w") as f:
        json.dump(analysis_output, f, indent=2)

    print("\n✓ Detailed analysis saved to: results/feature_importance_detailed.json")
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
