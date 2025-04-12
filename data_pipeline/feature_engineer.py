import numpy as np
import pandas as pd

class FeatureEngineer:
    """Class to create or transform features (e.g., log-transform, feature engineering)."""

    def __init__(self):
        pass

    def log_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Applies logarithmic transformations to the specified columns.

        Args:
            df (pd.DataFrame): DataFrame on which to apply transformations.
            columns (list): List of column names to log-transform.

        Returns:
            pd.DataFrame: The DataFrame with new log-transformed columns.
        """
        offset = 1e-6
        for col in columns:
            new_col = f"log_{col}"
            df[new_col] = np.log(df[col] + offset)
        return df

    def add_total_morosidad(self, df: pd.DataFrame,
                            col_30: str,
                            col_60: str,
                            col_90: str,
                            new_col: str = "Total_Morosidad") -> pd.DataFrame:
        """Creates the 'Total_Morosidad' variable as the sum of three specified columns.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            col_30 (str): Column name for 30-59 days past due.
            col_60 (str): Column name for 60-89 days past due.
            col_90 (str): Column name for 90+ days past due.
            new_col (str, optional): Name for the new column. Defaults to "Total_Morosidad".

        Returns:
            pd.DataFrame: The DataFrame with the new 'Total_Morosidad' column.
        """
        df[new_col] = df[col_30] + df[col_60] + df[col_90]
        return df

    def add_income_to_debt(self, df: pd.DataFrame,
                           log_income_col: str,
                           log_debt_col: str,
                           new_col: str = "Income_to_Debt") -> pd.DataFrame:
        """Creates the 'Income_to_Debt' variable as the ratio of log-transformed income to log-transformed debt.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            log_income_col (str): Column name for log-transformed income.
            log_debt_col (str): Column name for log-transformed debt.
            new_col (str, optional): Name for the new column. Defaults to "Income_to_Debt".

        Returns:
            pd.DataFrame: The DataFrame with the new 'Income_to_Debt' column.
        """
        offset = 1e-6
        df[new_col] = df[log_income_col] / (df[log_debt_col] + offset)
        return df
