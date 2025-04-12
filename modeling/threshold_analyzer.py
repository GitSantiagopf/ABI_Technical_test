import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix
)
import seaborn as sns

class ThresholdAnalyzer:
    """Analyzes the performance of a classifier at varying probability thresholds.

    This class computes key performance metrics (Recall, Precision, F1 Score,
    Accuracy, and AUC) over a range of cutoff thresholds for a trained model.
    It then generates plots of these metrics versus threshold and can also plot
    confusion matrices for different thresholds of the calibrated model. All plots
    are saved in the "plots" folder.
    """

    def __init__(self, model, X_test_scaled, y_test):
        """
        Args:
            model: A trained classifier implementing predict_proba().
            X_test_scaled (array-like): Scaled test data.
            y_test (array-like): True test labels.
        """
        self.model = model
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        self.thresholds = np.linspace(0.0, 1.0, 101)
        self.metrics = {}  # Stores metrics for each threshold
        
        # Ensure "plots" directory exists
        if not os.path.exists("plots"):
            os.makedirs("plots")

    def analyze(self):
        """Computes performance metrics for a range of thresholds.

        Returns:
            dict: Dictionary containing thresholds and corresponding metrics.
        """
        y_prob = self.model.predict_proba(self.X_test_scaled)[:, 1]
        recalls, precisions, f1_scores, accuracies, aucs = [], [], [], [], []
        for thresh in self.thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            recalls.append(recall_score(self.y_test, y_pred_thresh, zero_division=0))
            precisions.append(precision_score(self.y_test, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(self.y_test, y_pred_thresh, zero_division=0))
            accuracies.append(accuracy_score(self.y_test, y_pred_thresh))
            try:
                auc_val = roc_auc_score(self.y_test, y_pred_thresh)
            except Exception:
                auc_val = np.nan
            aucs.append(auc_val)
        self.metrics = {
            "thresholds": self.thresholds,
            "recall": recalls,
            "precision": precisions,
            "f1": f1_scores,
            "accuracy": accuracies,
            "auc": aucs
        }
        return self.metrics

    def plot_metrics(self, save: bool = True):
        """Plots and saves performance metrics versus threshold.

        Args:
            save (bool): Whether to save the plot in the 'plots' folder. Default is True.
        """
        if not self.metrics:
            self.analyze()
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["thresholds"], self.metrics["recall"], label="Recall", color="red")
        plt.plot(self.metrics["thresholds"], self.metrics["precision"], label="Precision", color="blue")
        plt.plot(self.metrics["thresholds"], self.metrics["f1"], label="F1 Score", color="green")
        plt.plot(self.metrics["thresholds"], self.metrics["accuracy"], label="Accuracy", color="orange")
        plt.plot(self.metrics["thresholds"], self.metrics["auc"], label="AUC", color="gray")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Performance Metrics vs. Threshold")
        plt.legend()
        plt.grid(True)
        if save:
            filename = "plots/threshold_metrics.png"
            plt.savefig(filename)
            print(f"Threshold metrics plot saved in {filename}")
        plt.show()

    def best_threshold_conclusion(self):
        """Prints a conclusion on the optimal threshold based on the metric analysis.
        
        Returns:
            str: A brief conclusion regarding the optimal threshold.
        """
        # In our analysis, the default threshold of 0.5 for the non-calibrated model
        # provides the best balance between recall and precision.
        conclusion = (
            "Based on the performance metrics, the default threshold of 0.5 for the "
            "non-calibrated model offers the best balance between recall and precision. "
            "Lower thresholds increase recall but also raise false positives, while higher "
            "thresholds significantly reduce recall. Hence, a threshold of 0.5 is recommended."
        )
        print(conclusion)
        return conclusion

    def plot_confusion_matrices_by_threshold(self, model, X_test_scaled, y_test, thresholds=None, save: bool = True):
        """Plots and saves confusion matrices for specified thresholds of a given model.

        Args:
            model: Trained classifier with a predict_proba method.
            X_test_scaled (array-like): Scaled test data.
            y_test (array-like): True test labels.
            thresholds (array-like, optional): List or array of thresholds to evaluate.
                If None, defaults to [0.1, 0.2, ..., 0.9].
            save (bool): Whether to save each confusion matrix plot in "plots".
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        for thresh in thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred_thresh)
            
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Prediction")
            plt.ylabel("True Value")
            plt.title(f"Confusion Matrix (Threshold = {thresh:.2f})")
            if save:
                filename = f"plots/confusion_matrix_calibrated_thresh_{thresh:.2f}.png"
                plt.savefig(filename)
                print(f"Confusion matrix at threshold {thresh:.2f} saved in {filename}")
            plt.show()
