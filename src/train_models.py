"""
Train classification models to predict task type from attention patterns.
Tests H4: Can we predict task type from attention features with >70% accuracy?
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class TaskClassifier:
    """Train and evaluate models for task classification"""

    def __init__(self, df, test_size=0.2, random_state=42):
        """
        Initialize classifier

        Args:
            df: DataFrame with attention features
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare features and labels"""
        print("Preparing data...")

        # Get feature columns
        self.feature_cols = [
            c for c in self.df.columns if c not in ["task_type", "prompt", "n_tokens"]
        ]

        # Features and labels
        X = self.df[self.feature_cols].values
        y = self.df["task_type"].values

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✓ Data prepared:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Classes: {len(self.label_encoder.classes_)}")
        print(f"  Class names: {self.label_encoder.classes_}")

    def train_logistic_regression(self):
        """Train Logistic Regression (baseline)"""
        print("\n" + "=" * 70)
        print("LOGISTIC REGRESSION")
        print("=" * 70)

        model = LogisticRegression(
            max_iter=1000, random_state=self.random_state, solver="lbfgs"
        )

        # Train
        print("Training...")
        model.fit(self.X_train_scaled, self.y_train)

        # Predict
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)

        # Evaluate
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)

        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train_scaled, self.y_train, cv=5, scoring="accuracy"
        )
        print(
            f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

        # Classification report
        print("\nClassification Report (Test Set):")
        print(
            classification_report(
                self.y_test,
                y_pred_test,
                target_names=self.label_encoder.classes_,
                digits=4,
            )
        )

        # Store results
        self.results["logistic_regression"] = {
            "model": model,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "predictions": y_pred_test,
        }

        return model

    def train_random_forest(self):
        """Train Random Forest"""
        print("\n" + "=" * 70)
        print("RANDOM FOREST")
        print("=" * 70)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Train
        print("Training...")
        model.fit(self.X_train_scaled, self.y_train)

        # Predict
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)

        # Evaluate
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)

        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train_scaled, self.y_train, cv=5, scoring="accuracy"
        )
        print(
            f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

        # Classification report
        print("\nClassification Report (Test Set):")
        print(
            classification_report(
                self.y_test,
                y_pred_test,
                target_names=self.label_encoder.classes_,
                digits=4,
            )
        )

        # Feature importance
        print("\nTop 20 Most Important Features:")
        feature_importance = pd.DataFrame(
            {"feature": self.feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(feature_importance.head(20).to_string(index=False))

        # Store results
        self.results["random_forest"] = {
            "model": model,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "predictions": y_pred_test,
            "feature_importance": feature_importance.head(20).to_dict("records"),
        }

        return model

    def train_svm(self):
        """Train Support Vector Machine"""
        print("\n" + "=" * 70)
        print("SUPPORT VECTOR MACHINE (SVM)")
        print("=" * 70)

        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=self.random_state,
            probability=True,
        )

        # Train
        print("Training...")
        model.fit(self.X_train_scaled, self.y_train)

        # Predict
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)

        # Evaluate
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)

        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train_scaled, self.y_train, cv=5, scoring="accuracy"
        )
        print(
            f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

        # Classification report
        print("\nClassification Report (Test Set):")
        print(
            classification_report(
                self.y_test,
                y_pred_test,
                target_names=self.label_encoder.classes_,
                digits=4,
            )
        )

        # Store results
        self.results["svm"] = {
            "model": model,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "predictions": y_pred_test,
        }

        return model

    def train_neural_network(self):
        """Train Multi-Layer Perceptron"""
        print("\n" + "=" * 70)
        print("NEURAL NETWORK (MLP)")
        print("=" * 70)

        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

        # Train
        print("Training...")
        model.fit(self.X_train_scaled, self.y_train)

        # Predict
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)

        # Evaluate
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)

        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Iterations: {model.n_iter_}")

        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train_scaled, self.y_train, cv=5, scoring="accuracy"
        )
        print(
            f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

        # Classification report
        print("\nClassification Report (Test Set):")
        print(
            classification_report(
                self.y_test,
                y_pred_test,
                target_names=self.label_encoder.classes_,
                digits=4,
            )
        )

        # Store results
        self.results["neural_network"] = {
            "model": model,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "predictions": y_pred_test,
            "n_iterations": int(model.n_iter_),
        }

        return model

    def plot_confusion_matrices(
        self, save_path="./results/visualizations/confusion_matrices.png"
    ):
        """Plot confusion matrices for all models"""
        print("\nGenerating confusion matrices...")

        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result["predictions"])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_,
                ax=axes[idx],
                cbar_kws={"label": "Count"},
            )

            axes[idx].set_title(
                f"{model_name.replace('_', ' ').title()}\nAccuracy: {result['test_accuracy']:.3f}",
                fontweight="bold",
                fontsize=12,
            )
            axes[idx].set_xlabel("Predicted", fontweight="bold")
            axes[idx].set_ylabel("True", fontweight="bold")
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved to: {save_path}")
        plt.close()

    def plot_model_comparison(
        self, save_path="./results/visualizations/model_comparison.png"
    ):
        """Compare model performance"""
        print("\nGenerating model comparison plot...")

        models = []
        test_accs = []
        cv_means = []
        cv_stds = []

        for model_name, result in self.results.items():
            models.append(model_name.replace("_", " ").title())
            test_accs.append(result["test_accuracy"])
            cv_means.append(result["cv_mean"])
            cv_stds.append(result["cv_std"])

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            test_accs,
            width,
            label="Test Accuracy",
            color="steelblue",
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            cv_means,
            width,
            label="CV Mean Accuracy",
            yerr=cv_stds,
            capsize=5,
            color="coral",
            edgecolor="black",
        )

        # Add 70% threshold line (H4)
        ax.axhline(
            y=0.70,
            color="red",
            linestyle="--",
            linewidth=2,
            label="H4 Threshold (70%)",
            alpha=0.7,
        )

        ax.set_xlabel("Model", fontweight="bold", fontsize=12)
        ax.set_ylabel("Accuracy", fontweight="bold", fontsize=12)
        ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved to: {save_path}")
        plt.close()

    def generate_report(self, output_file="./results/classification_report.json"):
        """Generate comprehensive classification report"""

        print("\n" + "=" * 70)
        print("GENERATING CLASSIFICATION REPORT")
        print("=" * 70)

        # Summary
        summary = {
            "dataset_info": {
                "n_train": len(self.X_train),
                "n_test": len(self.X_test),
                "n_features": len(self.feature_cols),
                "n_classes": len(self.label_encoder.classes_),
                "classes": self.label_encoder.classes_.tolist(),
            },
            "models": {},
        }

        # Add model results (exclude sklearn objects)
        for model_name, result in self.results.items():
            summary["models"][model_name] = {
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "cv_mean": result["cv_mean"],
                "cv_std": result["cv_std"],
            }

            # Add feature importance if available
            if "feature_importance" in result:
                summary["models"][model_name]["top_features"] = result[
                    "feature_importance"
                ]

        # H4 evaluation
        best_model = max(self.results.items(), key=lambda x: x[1]["test_accuracy"])
        best_acc = best_model[1]["test_accuracy"]

        summary["h4_evaluation"] = {
            "threshold": 0.70,
            "best_model": best_model[0],
            "best_accuracy": best_acc,
            "h4_supported": best_acc >= 0.70,
        }

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n✓ Report saved to: {output_file}")

        # Print H4 result
        print("\n" + "=" * 70)
        print("HYPOTHESIS 4 EVALUATION")
        print("=" * 70)
        print(f"Best Model: {best_model[0].replace('_', ' ').title()}")
        print(f"Test Accuracy: {best_acc:.4f}")
        print("Threshold: 0.70")

        if best_acc >= 0.70:
            print(
                "\n✓ H4 SUPPORTED: Task type can be predicted from attention patterns with >70% accuracy"
            )
        else:
            print(
                f"\n✗ H4 NOT SUPPORTED: Best accuracy ({best_acc:.4f}) below 70% threshold"
            )
        print("=" * 70)

        return summary


def main():
    """Main execution function"""

    import os

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_file = os.path.join(project_dir, "data", "processed", "attention_dataset.csv")

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} samples")

    # Initialize classifier
    classifier = TaskClassifier(df)

    # Train all models
    classifier.train_logistic_regression()
    classifier.train_random_forest()
    classifier.train_svm()
    classifier.train_neural_network()

    # Generate visualizations
    classifier.plot_confusion_matrices()
    classifier.plot_model_comparison()

    # Generate report
    classifier.generate_report()

    print("\n✓ Model training and evaluation complete!")


if __name__ == "__main__":
    main()
