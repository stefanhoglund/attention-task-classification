"""
Statistical hypothesis testing for attention pattern analysis.
Tests H1, H2, H3, H4 with proper statistical rigor.
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, ttest_ind

warnings.filterwarnings("ignore")


class HypothesisTester:
    """Conduct statistical tests for all hypotheses"""

    def __init__(self, df):
        """
        Initialize with dataset

        Args:
            df: DataFrame with attention features
        """
        self.df = df
        self.results = {}

        # Get feature columns
        self.feature_cols = [
            c for c in df.columns if c not in ["task_type", "prompt", "n_tokens"]
        ]

        # Get feature type columns
        self.entropy_cols = [c for c in self.feature_cols if c.startswith("entropy")]
        self.induction_cols = [
            c for c in self.feature_cols if c.startswith("induction")
        ]
        self.spread_cols = [c for c in self.feature_cols if c.startswith("spread")]
        self.self_attn_cols = [
            c for c in self.feature_cols if c.startswith("self_attn")
        ]

        # Calculate aggregate metrics
        self.df["mean_entropy"] = self.df[self.entropy_cols].mean(axis=1)
        self.df["mean_induction"] = self.df[self.induction_cols].mean(axis=1)
        self.df["mean_spread"] = self.df[self.spread_cols].mean(axis=1)
        self.df["mean_self_attn"] = self.df[self.self_attn_cols].mean(axis=1)

        print("HypothesisTester initialized")
        print(f"Dataset: {len(self.df)} samples, {len(self.feature_cols)} features")

    def test_h1_task_differences(self, alpha=0.05):
        """
        H1: Different task categories produce significantly different attention patterns

        Method: One-way ANOVA (or Kruskal-Wallis if non-normal) for each metric
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS 1: Task Categories Produce Different Attention Patterns")
        print("=" * 70)

        metrics = {
            "entropy": "mean_entropy",
            "induction": "mean_induction",
            "spread": "mean_spread",
            "self_attention": "mean_self_attn",
        }

        h1_results = {}

        for metric_name, column in metrics.items():
            print(f"\n{metric_name.upper()} Analysis:")
            print("-" * 50)

            # Group data by task
            groups = [
                self.df[self.df["task_type"] == task][column].values
                for task in self.df["task_type"].unique()
            ]

            # Check normality (Shapiro-Wilk test)
            normality_tests = [stats.shapiro(group) for group in groups]
            all_normal = all(p > alpha for _, p in normality_tests)

            # Choose test based on normality
            if all_normal and len(groups[0]) > 20:
                # Parametric: One-way ANOVA
                f_stat, p_value = f_oneway(*groups)
                test_used = "One-way ANOVA"
            else:
                # Non-parametric: Kruskal-Wallis
                h_stat, p_value = kruskal(*groups)
                test_used = "Kruskal-Wallis H-test"
                f_stat = h_stat

            # Calculate effect size (eta-squared)
            grand_mean = self.df[column].mean()
            ss_between = sum(
                len(group) * (group.mean() - grand_mean) ** 2 for group in groups
            )
            ss_total = sum((self.df[column] - grand_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            # Interpret effect size
            if eta_squared < 0.01:
                effect_size_interpretation = "negligible"
            elif eta_squared < 0.06:
                effect_size_interpretation = "small"
            elif eta_squared < 0.14:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"

            # Print results
            print(f"Test used: {test_used}")
            print(f"Test statistic: {f_stat:.4f}")
            print(f"P-value: {p_value:.6f}")
            print(f"Effect size (η²): {eta_squared:.4f} ({effect_size_interpretation})")

            if p_value < alpha:
                print(f"✓ SIGNIFICANT: Tasks differ in {metric_name} (p < {alpha})")
            else:
                print(
                    f"✗ NOT SIGNIFICANT: No difference in {metric_name} (p >= {alpha})"
                )

            # Descriptive statistics by task
            print("\nDescriptive Statistics:")
            desc_stats = self.df.groupby("task_type")[column].agg(
                ["mean", "std", "count"]
            )
            print(desc_stats.round(4))

            # Post-hoc pairwise comparisons (if significant)
            if p_value < alpha:
                print("\nPost-hoc Pairwise Comparisons (Bonferroni corrected):")
                tasks = self.df["task_type"].unique()
                n_comparisons = len(tasks) * (len(tasks) - 1) // 2
                bonferroni_alpha = alpha / n_comparisons

                significant_pairs = []
                for i, task1 in enumerate(tasks):
                    for task2 in tasks[i + 1 :]:
                        group1 = self.df[self.df["task_type"] == task1][column].values
                        group2 = self.df[self.df["task_type"] == task2][column].values

                        t_stat, p_val = ttest_ind(group1, group2)

                        if p_val < bonferroni_alpha:
                            diff = group1.mean() - group2.mean()
                            significant_pairs.append((task1, task2, diff, p_val))
                            print(
                                f"  {task1} vs {task2}: "
                                f"diff={diff:.4f}, p={p_val:.6f} *"
                            )

                if not significant_pairs:
                    print(
                        "  No significant pairwise differences after Bonferroni correction"
                    )

            # Store results
            h1_results[metric_name] = {
                "test_used": test_used,
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_squared),
                "effect_size": effect_size_interpretation,
                "significant": p_value < alpha,
                "descriptive_stats": desc_stats.to_dict(),
            }

        self.results["H1"] = h1_results

        # Overall H1 conclusion
        print("\n" + "=" * 70)
        print("H1 CONCLUSION:")
        n_significant = sum(1 for r in h1_results.values() if r["significant"])
        if n_significant >= 3:
            print(
                f"✓ H1 SUPPORTED: {n_significant}/4 metrics show significant differences"
            )
        else:
            print(f"✗ H1 NOT SUPPORTED: Only {n_significant}/4 metrics significant")
        print("=" * 70)

        return h1_results

    def test_h2_layer_specificity(self, alpha=0.05):
        """
        H2: Later layers (8-11) show more task-specific patterns than early layers (0-3)

        Method: Compare between-task variance in early vs late layers
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS 2: Later Layers Show More Task-Specific Patterns")
        print("=" * 70)

        early_layers = [0, 1, 2, 3]
        late_layers = [8, 9, 10, 11]

        # Calculate variance across tasks for each layer
        early_variances = []
        late_variances = []

        print("\nCalculating between-task variance by layer...")

        for layer in early_layers:
            # Get all features for this layer
            layer_cols = [c for c in self.entropy_cols if f"_L{layer}H" in c]

            # Calculate mean per task
            task_means = self.df.groupby("task_type")[layer_cols].mean()

            # Variance across tasks (higher = more task-specific)
            variance = task_means.var().mean()
            early_variances.append(variance)
            print(f"  Layer {layer}: variance = {variance:.6f}")

        print()

        for layer in late_layers:
            layer_cols = [c for c in self.entropy_cols if f"_L{layer}H" in c]
            task_means = self.df.groupby("task_type")[layer_cols].mean()
            variance = task_means.var().mean()
            late_variances.append(variance)
            print(f"  Layer {layer}: variance = {variance:.6f}")

        # Statistical test: Independent t-test
        t_stat, p_value = ttest_ind(late_variances, early_variances)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(early_variances) + np.var(late_variances)) / 2)
        cohens_d = (
            (np.mean(late_variances) - np.mean(early_variances)) / pooled_std
            if pooled_std > 0
            else 0
        )

        # Results
        print("\nStatistical Test: Independent t-test")
        print(f"Early layers mean variance: {np.mean(early_variances):.6f}")
        print(f"Late layers mean variance:  {np.mean(late_variances):.6f}")
        print(f"Difference: {np.mean(late_variances) - np.mean(early_variances):.6f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Cohen's d: {cohens_d:.4f}")

        # Interpretation
        if cohens_d < 0.2:
            effect_interpretation = "negligible"
        elif cohens_d < 0.5:
            effect_interpretation = "small"
        elif cohens_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        print(f"Effect size: {effect_interpretation}")

        h2_result = {
            "early_layers": early_layers,
            "late_layers": late_layers,
            "early_mean_variance": float(np.mean(early_variances)),
            "late_mean_variance": float(np.mean(late_variances)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size": effect_interpretation,
            "significant": p_value < alpha,
        }

        self.results["H2"] = h2_result

        # Conclusion
        print("\n" + "=" * 70)
        if p_value < alpha and t_stat > 0:
            print("✓ H2 SUPPORTED: Later layers are significantly more task-specific")
        else:
            print("✗ H2 NOT SUPPORTED: No significant difference in task-specificity")
        print("=" * 70)

        return h2_result

    def test_h3_entropy_task_uncertainty(self, alpha=0.05):
        """
        H3: Attention entropy correlates with task uncertainty
        Expected: creative > logical > factual/arithmetic

        Method: Ordered hypothesis testing
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS 3: Attention Entropy Correlates with Task Uncertainty")
        print("=" * 70)

        # Calculate mean entropy per task
        task_entropy = (
            self.df.groupby("task_type")["mean_entropy"]
            .mean()
            .sort_values(ascending=False)
        )

        print("\nMean Entropy by Task (ordered):")
        for task, entropy in task_entropy.items():
            print(f"  {task:25s}: {entropy:.6f}")

        # Expected ordering: creative > logical > factual ≈ arithmetic
        expected_order = {
            "creative_generation": 1,
            "logical_inference": 2,
            "factual_recall": 3,
            "arithmetic": 3,  # Same rank as factual
        }

        actual_ranks = task_entropy.rank(ascending=False).to_dict()

        print("\nExpected vs Actual Ranking:")
        print(f"{'Task':<25s} {'Expected':<10s} {'Actual':<10s}")
        print("-" * 50)
        for task in expected_order.keys():
            print(
                f"{task:<25s} {expected_order[task]:<10.0f} {actual_ranks[task]:<10.1f}"
            )

        # Specific pairwise tests
        print("\nPairwise Comparisons:")

        comparisons = [
            ("creative_generation", "logical_inference", "Creative > Logical"),
            ("creative_generation", "factual_recall", "Creative > Factual"),
            ("creative_generation", "arithmetic", "Creative > Arithmetic"),
            ("logical_inference", "factual_recall", "Logical > Factual"),
            ("logical_inference", "arithmetic", "Logical > Arithmetic"),
        ]

        comparison_results = []
        for task1, task2, description in comparisons:
            group1 = self.df[self.df["task_type"] == task1]["mean_entropy"].values
            group2 = self.df[self.df["task_type"] == task2]["mean_entropy"].values

            # One-tailed t-test (testing if task1 > task2)
            t_stat, p_value_two_tailed = ttest_ind(group1, group2)
            p_value = (
                p_value_two_tailed / 2 if t_stat > 0 else 1 - p_value_two_tailed / 2
            )

            mean_diff = group1.mean() - group2.mean()
            is_expected = mean_diff > 0
            is_significant = p_value < alpha

            comparison_results.append(
                {
                    "comparison": description,
                    "task1": task1,
                    "task2": task2,
                    "mean_diff": float(mean_diff),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "matches_expectation": is_expected,
                    "significant": is_significant,
                }
            )

            result_symbol = (
                "✓"
                if is_expected and is_significant
                else "✗"
                if not is_expected
                else "~"
            )
            print(
                f"  {result_symbol} {description}: diff={mean_diff:.4f}, "
                f"t={t_stat:.3f}, p={p_value:.4f}"
            )

        # Overall conclusion
        n_matching = sum(1 for r in comparison_results if r["matches_expectation"])
        n_significant = sum(
            1
            for r in comparison_results
            if r["significant"] and r["matches_expectation"]
        )

        h3_result = {
            "task_entropy_means": {k: float(v) for k, v in task_entropy.items()},
            "expected_order": expected_order,
            "actual_ranks": {k: float(v) for k, v in actual_ranks.items()},
            "comparisons": comparison_results,
            "n_matching_expectation": n_matching,
            "n_significant_matches": n_significant,
            "supported": n_significant >= 3,
        }

        self.results["H3"] = h3_result

        print("\n" + "=" * 70)
        if n_significant >= 3:
            print(
                f"✓ H3 SUPPORTED: {n_significant}/{len(comparisons)} expected patterns significant"
            )
        else:
            print(
                f"✗ H3 PARTIALLY SUPPORTED: {n_significant}/{len(comparisons)} significant"
            )
        print("=" * 70)

        return h3_result

    def generate_report(self, output_file="./results/statistical_analysis_report.json"):
        """Generate comprehensive report"""

        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE STATISTICAL REPORT")
        print("=" * 70)

        # Overall summary
        summary = {
            "dataset_info": {
                "n_samples": len(self.df),
                "n_features": len(self.feature_cols),
                "task_distribution": self.df["task_type"].value_counts().to_dict(),
            },
            "hypothesis_tests": self.results,
            "overall_conclusion": {
                "H1_supported": self.results.get("H1", {})
                .get("entropy", {})
                .get("significant", False),
                "H2_supported": self.results.get("H2", {}).get("significant", False),
                "H3_supported": self.results.get("H3", {}).get("supported", False),
            },
        }

        # Save to JSON
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n✓ Report saved to: {output_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        for hyp, supported in summary["overall_conclusion"].items():
            status = "✓ SUPPORTED" if supported else "✗ NOT SUPPORTED"
            print(f"{hyp}: {status}")
        print("=" * 70)

        return summary


def main():
    """Main execution function"""

    # Load data
    print("Loading dataset...")
    df = pd.read_csv("./data/processed/attention_dataset.csv")
    print(f"✓ Loaded {len(df)} samples")

    # Initialize tester
    tester = HypothesisTester(df)

    # Test all hypotheses
    tester.test_h1_task_differences()
    tester.test_h2_layer_specificity()
    tester.test_h3_entropy_task_uncertainty()

    # Generate report
    tester.generate_report()

    print("\n✓ Statistical analysis complete!")


if __name__ == "__main__":
    main()
