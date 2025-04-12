import pandas as pd

class DataCleaner:
    """Class responsible for cleaning and imputing missing values in a DataFrame."""
    
    def __init__(self):
        pass

    def dropna_in_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Drops rows with NaN values in the specified column.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
            column (str): The column name on which to drop NA values.

        Returns:
            pd.DataFrame: A DataFrame with rows missing values in the specified column removed.
        """
        initial_rows = df.shape[0]
        df_clean = df.dropna(subset=[column])
        removed = initial_rows - df_clean.shape[0]
        print(f"Dropped {removed} records due to missing '{column}'.")
        return df_clean

    def fillna_with_value(self, df: pd.DataFrame, column: str, value=0) -> pd.DataFrame:
        """Fills missing values in the specified column with a given value.

        Args:
            df (pd.DataFrame): The DataFrame to impute.
            column (str): The column name to impute.
            value (optional): The value to fill missing values with. Default is 0.

        Returns:
            pd.DataFrame: The DataFrame with missing values in the specified column replaced.
        """
        df[column] = df[column].fillna(value)
        print(f"Filled missing values in '{column}' with {value}.")
        return df
