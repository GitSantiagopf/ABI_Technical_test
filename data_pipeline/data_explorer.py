import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

class DataExplorer:
    """Class for data exploration and visualization methods."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Create the "plots" directory if it does not exist.
        if not os.path.exists("plots"):
            os.makedirs("plots")
    
    def info_data(self):
        """Prints general DataFrame information and descriptive statistics."""
        print("\nDataFrame Information:")
        print(self.df.info())
        print("\nDescriptive Statistics:")
        print(self.df.describe().T)

    def plot_distribution(self, col: str, apply_log: bool = False, save: bool = True):
        """Plots a histogram and KDE for a given column with an optional log-transform.
        
        Filters non-finite values and ensures at least two unique values exist before
        computing the KDE. Optionally saves the plot to the 'plots' folder.

        Args:
            col (str): The column name to plot.
            apply_log (bool, optional): If True, applies a log transformation. Default is False.
            save (bool, optional): If True, saves the plot image in "plots". Default is True.
        """
        plt.figure(figsize=(10, 6))
        offset = 1e-6
        # Replace infs with NaN and drop NaN values.
        values = self.df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if apply_log:
            values = np.log(values + offset)
            label = f'Log of {col}'
        else:
            label = col
        
        if len(np.unique(values)) < 2:
            print(f"Not enough unique values to compute KDE for '{col}'. Plotting only the histogram.")
            plt.hist(values, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        else:
            plt.hist(values, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
            try:
                kde = gaussian_kde(values)
                xs = np.linspace(values.min(), values.max(), 200)
                plt.plot(xs, kde(xs), color='red', linewidth=2, label='KDE')
            except Exception as e:
                print(f"KDE could not be computed for '{col}': {e}")
        
        plt.title(f"Distribution of {col}{' (Log)' if apply_log else ''}")
        plt.xlabel(label)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        
        if save:
            filename = f"plots/{col}{'_log' if apply_log else ''}.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")
        plt.show()

    def correlation_heatmap(self, columns: list, save: bool = True):
        """Generates a correlation heatmap for the specified columns and optionally saves it.

        Args:
            columns (list): A list of column names to include in the heatmap.
            save (bool, optional): If True, saves the heatmap image in "plots". Default is True.
        """
        corr_matrix = self.df[columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        if save:
            filename = "plots/correlation_heatmap.png"
            plt.savefig(filename)
            print(f"Heatmap saved as {filename}")
        plt.show()
