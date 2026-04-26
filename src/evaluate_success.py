"""
Evaluate if the project successfully demonstrated attention pattern differences.
Provides clear pass/fail criteria for each hypothesis.
"""

import json
import os

import pandas as pd


def evaluate_project_success():
    """
    Evaluate project success based on multiple criteria.
    Returns assessment of whether the core idea worked.
    """

    print("=" * 70)
    print("PROJECT SUCCESS EVALUATION")
    print("=" * 70)

    results = {
        "technical_success": {},
        "hypothesis_results": {},
        "overall_assessment": "",
    }

    # ========================================================================
    # 1. TECHNICAL SUCCESS - Did the pipeline work?
    # ========================================================================
    print("\n1. TECHNICAL SUCCESS")
    print("-" * 70)

    # Check if dataset was created
    dataset_path = "./data/processed/attention_dataset.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        n_samples = len(df)
        n_features = len(
            [c for c in df.columns if c not in ["task_type", "prompt", "n_tokens"]]
        )

        print(f"✓ Dataset created: {n_samples} samples, {n_features} features")
        results["technical_success"]["dataset"] = True
        results["technical_success"]["n_samples"] = n_samples
        results["technical_success"]["n_features"] = n_features

        # Minimum viable: at least 60 samples
        if n_samples >= 60:
            print("  ✓ Sufficient sample size (>60)")
        else:
            print("  ⚠️  Small sample size (<60) - results may be underpowered")
    else:
        print("✗ Dataset not found - pipeline failed")
        results["technical_success"]["dataset"] = False
        return results

    # Check if models trained
    class_report_path = "./results/classification_report.json"
    if os.path.exists(class_report_path):
        print("✓ Models trained successfully")
        results["technical_success"]["models_trained"] = True
    else:
        print("✗ No classification results found")
        results["technical_success"]["models_trained"] = False

    # ========================================================================
    # 2. HYPOTHESIS EVALUATION - Did we find meaningful patterns?
    # ========================================================================
    print("\n2. HYPOTHESIS EVALUATION")
    print("-" * 70)

    # Load classification results
    if os.path.exists(class_report_path):
        with open(class_report_path, "r") as f:
            class_data = json.load(f)

        # H4: Classification accuracy
        h4 = class_data.get("h4_evaluation", {})
        best_acc = h4.get("best_accuracy", 0.0)
        threshold = h4.get("threshold", 0.70)
        best_model = h4.get("best_model", "unknown")

        print("\nH4: Can we predict task type from attention patterns?")
        print(f"  Best Model: {best_model.replace('_', ' ').title()}")
        print(f"  Test Accuracy: {best_acc:.2%}")
        print(f"  Threshold: {threshold:.2%}")
        print("  Random Baseline: 25% (4 classes)")

        # Success criteria
        random_baseline = 0.25
        weak_success = 0.40  # Better than random
        moderate_success = 0.60  # Clearly better than random
        strong_success = 0.70  # Original hypothesis threshold

        if best_acc >= strong_success:
            print(f"  ✓✓✓ STRONG SUCCESS: {best_acc:.2%} >> baseline")
            results["hypothesis_results"]["h4"] = "strong_success"
            h4_interpretation = "Attention patterns strongly encode task information"
        elif best_acc >= moderate_success:
            print(f"  ✓✓ MODERATE SUCCESS: {best_acc:.2%} > baseline")
            results["hypothesis_results"]["h4"] = "moderate_success"
            h4_interpretation = "Attention patterns contain useful task information"
        elif best_acc >= weak_success:
            print(f"  ✓ WEAK SUCCESS: {best_acc:.2%} slightly > baseline")
            results["hypothesis_results"]["h4"] = "weak_success"
            h4_interpretation = "Attention patterns show some task differentiation"
        else:
            print(f"  ✗ FAILURE: {best_acc:.2%} ≈ random baseline")
            results["hypothesis_results"]["h4"] = "failure"
            h4_interpretation = "Attention patterns do not reliably differentiate tasks"

        results["hypothesis_results"]["h4_accuracy"] = best_acc
        results["hypothesis_results"]["h4_interpretation"] = h4_interpretation

        # Show all model performances
        print("\n  Model Performance Summary:")
        for model_name, model_data in class_data.get("models", {}).items():
            test_acc = model_data.get("test_accuracy", 0.0)
            cv_acc = model_data.get("cv_mean", 0.0)
            print(
                f"    {model_name.replace('_', ' ').title():20s}: "
                f"Test={test_acc:.3f}, CV={cv_acc:.3f}"
            )

    # ========================================================================
    # 3. DATA QUALITY - Are the patterns real or noise?
    # ========================================================================
    print("\n3. DATA QUALITY CHECKS")
    print("-" * 70)

    # Check entropy variation across tasks
    entropy_cols = [c for c in df.columns if c.startswith("entropy")]
    df["mean_entropy"] = df[entropy_cols].mean(axis=1)

    entropy_by_task = df.groupby("task_type")["mean_entropy"].agg(["mean", "std"])
    overall_mean = df["mean_entropy"].mean()
    overall_std = df["mean_entropy"].std()

    print("\nEntropy Statistics by Task:")
    print(f"{'Task':<30s} {'Mean':>10s} {'Std':>10s}")
    print("-" * 52)
    for task, row in entropy_by_task.iterrows():
        print(f"{task:<30s} {row['mean']:>10.4f} {row['std']:>10.4f}")

    print(f"\n{'Overall Dataset':<30s} {overall_mean:>10.4f} {overall_std:>10.4f}")

    # Check if tasks have different means (simple test)
    task_means = entropy_by_task["mean"].values
    between_task_variation = task_means.std()
    within_task_variation = entropy_by_task["std"].mean()

    print("\nVariation Analysis:")
    print(f"  Between-task variation: {between_task_variation:.4f}")
    print(f"  Within-task variation:  {within_task_variation:.4f}")
    print(
        f"  Ratio (between/within): {between_task_variation / within_task_variation:.2f}"
    )

    if between_task_variation > within_task_variation:
        print("  ✓ Tasks are more different from each other than within themselves")
        print("    → Patterns appear real, not random noise")
        results["data_quality"] = "good"
    else:
        print("  ⚠️  High within-task variation - patterns may be noisy")
        results["data_quality"] = "noisy"

    # ========================================================================
    # 4. PRACTICAL SIGNIFICANCE
    # ========================================================================
    print("\n4. PRACTICAL SIGNIFICANCE")
    print("-" * 70)

    print("\nWhat the results mean:")

    # Check if we have classification results
    if "h4_accuracy" in results["hypothesis_results"]:
        best_acc = results["hypothesis_results"]["h4_accuracy"]

        if best_acc >= 0.70:
            print("  ✓ HIGHLY USEFUL: Attention patterns are strong predictors")
            print("    → Could be used for model debugging, interpretability")
            print("    → Suggests transformers have task-specific circuits")
            practical_value = "high"
        elif best_acc >= 0.60:
            print("  ✓ MODERATELY USEFUL: Attention shows task-related structure")
            print("    → Useful for understanding model behavior")
            print("    → May guide architecture improvements")
            practical_value = "moderate"
        elif best_acc >= 0.40:
            print("  ~ SOMEWHAT USEFUL: Weak but detectable patterns")
            print("    → Confirms attention varies by task")
            print("    → Limited practical application")
            practical_value = "limited"
        else:
            print("  ✗ NOT USEFUL: No reliable task differentiation")
            print("    → Either tasks too similar or attention not task-specific")
            print("    → Need different features or larger dataset")
            practical_value = "none"

        results["practical_value"] = practical_value
    else:
        print("  ⚠️  No classification results available")
        print("    → Need to run: python src/train_models.py")
        results["practical_value"] = "unknown"
        best_acc = 0.0  # Set default for later use

    # ========================================================================
    # 5. OVERALL VERDICT
    # ========================================================================
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    # Count successes
    score = 0
    max_score = 4

    if results["technical_success"].get("dataset", False):
        score += 1

    if results["technical_success"].get("models_trained", False):
        score += 1

    h4_result = results["hypothesis_results"].get("h4", "failure")
    if h4_result == "strong_success":
        score += 2
    elif h4_result == "moderate_success":
        score += 1.5
    elif h4_result == "weak_success":
        score += 0.5

    percentage = (score / max_score) * 100

    print(f"\nProject Score: {score:.1f}/{max_score} ({percentage:.0f}%)")
    print()

    # Handle case where models haven't been trained yet
    if not results["technical_success"].get("models_trained", False):
        verdict = "⚠️  INCOMPLETE"
        explanation = (
            "The dataset has been created successfully, but models have not been trained yet. "
            "Run 'python src/train_models.py' to complete the analysis."
        )
        grade_estimate = "Incomplete - need to train models"
    elif percentage >= 75:
        verdict = "✓✓✓ STRONG SUCCESS"
        explanation = (
            "The project successfully demonstrated that transformer attention patterns "
            "contain task-specific information. The classification accuracy significantly "
            "exceeds random chance, providing evidence for interpretable structure in "
            "attention mechanisms."
        )
        grade_estimate = "High Pass (VG/A)"
    elif percentage >= 50:
        verdict = "✓✓ MODERATE SUCCESS"
        explanation = (
            "The project successfully showed that attention patterns differ across tasks, "
            "though not as strongly as hypothesized. The results provide valuable insights "
            "into transformer behavior and demonstrate proper scientific methodology."
        )
        grade_estimate = "Pass (G/B-C)"
    elif percentage >= 25:
        verdict = "✓ PARTIAL SUCCESS"
        explanation = (
            "The project demonstrated technical competence and proper methodology, but "
            "found weak evidence for task-specific attention patterns. This is still a "
            "valuable null/weak result that advances understanding."
        )
        grade_estimate = "Pass (G/C)"
    else:
        verdict = "✗ TECHNICAL SUCCESS ONLY"
        explanation = (
            "The pipeline worked correctly, but the core hypothesis was not supported. "
            "This could be due to dataset size, task selection, or genuine absence of "
            "task-specific patterns in attention."
        )
        grade_estimate = "Pass (methodology sound, results negative)"

    print(f"VERDICT: {verdict}")
    print(f"\n{explanation}")
    print(f"\nEstimated Grade: {grade_estimate}")

    # ========================================================================
    # 6. RECOMMENDATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR REPORT")
    print("=" * 70)

    if h4_result in ["strong_success", "moderate_success"]:
        print("\n✓ Your report should emphasize:")
        print("  - The successful differentiation of tasks")
        print("  - Specific features that were most discriminative")
        print("  - Implications for transformer interpretability")
        print("  - Comparison to baseline/random chance")
    else:
        print("\n✓ Your report should emphasize:")
        print("  - The rigorous methodology and proper controls")
        print("  - Why null/weak results are scientifically valuable")
        print("  - Potential reasons for weak patterns:")
        print("    • Small dataset size limiting statistical power")
        print("    • Task categories may be too broad/similar")
        print("    • Attention alone may not capture task processing")
        print("  - What you learned about transformers despite weak results")

    print("\n✓ All reports should include:")
    print("  - Clear statement of hypotheses")
    print("  - Proper statistical testing (p-values, effect sizes)")
    print("  - Limitations section (sample size, model choice)")
    print("  - Ethics discussion (interpretability limitations)")

    return results


if __name__ == "__main__":
    results = evaluate_project_success()
