import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

class DataTransformer:
    """Class for scaling data and preserving the scaler.

    Uses RobustScaler to reduce the impact of outliers.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Fits the scaler on the provided features and transforms the data.

        Args:
            df (pd.DataFrame): The DataFrame containing the features.
            feature_cols (list): List of column names to scale.

        Returns:
            np.ndarray: The scaled features as an array.
        """
        data = df[feature_cols].values
        data_scaled = self.scaler.fit_transform(data)
        return data_scaled

    def transform(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Transforms the provided data using the previously fitted scaler.

        Args:
            df (pd.DataFrame): The DataFrame containing the features.
            feature_cols (list): List of column names to scale.

        Returns:
            np.ndarray: The scaled features as an array.
        """
        data = df[feature_cols].values
        return self.scaler.transform(data)
