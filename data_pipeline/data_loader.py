import pandas as pd

class DataLoader:
    """Class responsible for loading data from CSV files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load_data(self) -> pd.DataFrame:
        """Loads a CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: Data loaded from the CSV file.
        """
        df = pd.read_csv(self.filepath)
        return df
