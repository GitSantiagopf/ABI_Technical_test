import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score,
                             classification_report, confusion_matrix, roc_curve)
from sklearn.calibration import calibration_curve


class ModelEvaluator:
    """Evaluates models on test data and plots evaluation metrics.

    This class computes metrics such as Accuracy, ROC AUC, and Recall for a set of models.
    It also provides methods for plotting and saving confusion matrices, calibration curves,
    and ROC curves in the 'plots' folder.
    """
    def __init__(self):
        self.results = {}
        if not os.path.exists("plots"):
            os.makedirs("plots")

    def evaluate_models(self, models_dict, X_test_scaled, y_test, positive_label=1):
        """Evaluates each model and prints a classification report.

        Args:
            models_dict (dict): Dictionary mapping model names to trained model objects.
            X_test_scaled (array-like): Scaled test data.
            y_test (array-like): True test labels.
            positive_label (int): Label of the positive class. Default is 1.

        Returns:
            A DataFrame containing Accuracy, ROC AUC, and Recall for each model.
        """
        for name, model in models_dict.items():
            print(f"\nEvaluate model: {name}")
            y_pred = model.predict(X_test_scaled)
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            rec = recall_score(y_test, y_pred, pos_label=positive_label)

            self.results[name] = {
                "Accuracy": acc,
                "ROC AUC": auc,
                "Recall (Default)": rec
            }

            print("Classification report:")
            print(classification_report(y_test, y_pred))
            print("-" * 50)
        
        return pd.DataFrame(self.results).T

    def plot_confusion_matrices(self, models_dict, X_test_scaled, y_test, save: bool = True):
        """Plots and saves confusion matrices for each model.

        Args:
            models_dict (dict): Dictionary mapping model names to model objects.
            X_test_scaled (array-like): Scaled test data.
            y_test (array-like): True test labels.
            save (bool): If True, saves the plots to 'plots'. Default is True.
        """
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.title(f"confusion matrix - {model_name}")
            if save:
                filename = f"plots/confusion_matrix_{model_name.replace(' ', '_')}.png"
                plt.savefig(filename)
                print(f"confusion matrix saved in {filename}")
            plt.show()

    def plot_calibration_curve(self, model, X_train_scaled, y_train, model_name="Model", save: bool = True):
        """Plots and saves the calibration curve for a given model.

        Args:
            model: Trained classifier implementing predict_proba.
            X_train_scaled (array-like): Scaled training data.
            y_train (array-like): True training labels.
            model_name (str): Name for the model, used in the plot title and filename.
            save (bool): If True, saves the plot to 'plots'. Default is True.
        """
        prob_pred, prob_true = calibration_curve(y_train, model.predict_proba(X_train_scaled)[:, 1], n_bins=10)

        plt.figure(figsize=(8,6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{model_name}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibración perfecta')
        plt.xlabel("Probabilidad media predicha")
        plt.ylabel("Fracción de positivos")
        plt.title(f"Curva de Calibración - {model_name}")
        plt.legend()
        plt.grid(True)

        if save:
            filename = f"plots/calibration_curve_{model_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            print(f"calibration curve saved in {filename}")

        plt.show()

    def plot_roc_curve(self, model, X_test_scaled, y_test, model_name="Model", save: bool = True):
        """Plots and saves the ROC curve for a given model.

        Args:
            model: Trained classifier implementing predict_proba.
            X_test_scaled (array-like): Scaled test data.
            y_test (array-like): True test labels.
            model_name (str): Name for the model, used in the plot title and filename.
            save (bool): If True, saves the plot to 'plots'. Default is True.
        """
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Línea Base')
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title(f"Curva ROC - {model_name}")
        plt.legend()
        plt.grid(True)

        if save:
            filename = f"plots/roc_curve_{model_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            print(f"ROC curve saved in {filename}")

        plt.show()
