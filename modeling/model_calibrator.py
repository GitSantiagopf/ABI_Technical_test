from sklearn.calibration import CalibratedClassifierCV

class ModelCalibrator:
    """Calibrates a model using isotonic or sigmoid methods.

    This class wraps a base classifier with a calibration step using cross-validation.
    """
    def __init__(self, base_model, method='isotonic'):
        self.base_model = base_model
        self.method = method
        self.calibrated_model = None

    def calibrate(self, X_train, y_train):
        """Calibrates the base model and returns the calibrated model.

        Args:
            X_train (array-like): Training data used for calibration.
            y_train (array-like): True labels for the training data.

        Returns:
            The calibrated model (CalibratedClassifierCV object).
        """
        self.calibrated_model = CalibratedClassifierCV(self.base_model,
                                                       method=self.method,
                                                       cv=5)
        self.calibrated_model.fit(X_train, y_train)
        return self.calibrated_model
