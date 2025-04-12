import pandas as pd
import numpy as np
import os

class Predictor:
    """Generates predictions and saves results to a CSV file.

    This class loads a test CSV file, applies the same transformations and scaling
    as used during training, predicts the probability for the positive class using
    the provided model, and finally exports the results (including an identifier if available)
    to an output CSV file.
    """
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols

    def predict_csv(self, input_csv: str, output_csv: str):
        """Generates predictions from an input CSV file and saves the results.

        The method loads the input CSV file, applies necessary transformations
        (e.g., log-transform on "MonthlyIncome", "DebtRatio", and "RevolvingUtilizationOfUnsecuredLines"),
        creates derived features such as "Total_Morosidad" and "Income_to_Debt", scales the data,
        and finally predicts the probability for the positive class. The results are stored in a CSV file.

        Args:
            input_csv (str): Path to the input CSV file.
            output_csv (str): Path to the output CSV file where predictions will be saved.
        """
        df_test = pd.read_csv(input_csv)
        
        if 'MonthlyIncome' in df_test.columns:
            df_test.dropna(subset=['MonthlyIncome'], inplace=True)
        
        if 'NumberOfDependents' in df_test.columns:
            df_test['NumberOfDependents'] = df_test['NumberOfDependents'].fillna(0)

        offset = 1e-6
        if 'MonthlyIncome' in df_test.columns:
            df_test['log_MonthlyIncome'] = np.log(df_test['MonthlyIncome'] + offset)
        if 'DebtRatio' in df_test.columns:
            df_test['log_DebtRatio'] = np.log(df_test['DebtRatio'] + offset)
        if 'RevolvingUtilizationOfUnsecuredLines' in df_test.columns:
            df_test['log_RevolvingUtilizationOfUnsecuredLines'] = np.log(
                df_test['RevolvingUtilizationOfUnsecuredLines'] + offset
            )

        if all(col in df_test.columns for col in [
                'NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate',
                'NumberOfTime60-89DaysPastDueNotWorse']):
            df_test['Total_Morosidad'] = (
                df_test['NumberOfTime30-59DaysPastDueNotWorse'] +
                df_test['NumberOfTimes90DaysLate'] +
                df_test['NumberOfTime60-89DaysPastDueNotWorse']
            )
        if all(col in df_test.columns for col in ['log_MonthlyIncome', 'log_DebtRatio']):
            df_test['Income_to_Debt'] = (
                df_test['log_MonthlyIncome'] /
                (df_test['log_DebtRatio'] + offset)
            )

        X_test_final = df_test[self.feature_cols].copy()
        X_test_final_scaled = self.scaler.transform(X_test_final)

        y_prob = self.model.predict_proba(X_test_final_scaled)[:, 1]

        if 'ID' in df_test.columns:
            results_df = df_test[['ID']].copy()
        else:
            results_df = pd.DataFrame({'ID': df_test.index+1})

        results_df['RiskProbability'] = y_prob
        results_df.to_csv(output_csv, index=False)
        print(f"saved predictions in {output_csv}")
