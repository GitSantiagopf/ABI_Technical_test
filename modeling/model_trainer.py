import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    """Trains classification models using GridSearchCV, optimizing for recall.

    This class conducts a grid search over the provided hyperparameter grid
    for a given model using cross-validation, returning the best estimator found.
    """
    def __init__(self, scoring='recall'):
        self.scoring = scoring

    def run_grid_search(self, model, param_grid, X_train, y_train):
        """Performs grid search with cross-validation on the input model.

        Args:
            model: The estimator to train.
            param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try.
            X_train (array-like): The training data.
            y_train (array-like): The target training labels.

        Returns:
            The best estimator found by GridSearchCV.
        """
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring=self.scoring,
                            cv=5,
                            n_jobs=-1,
                            verbose=1)
        grid.fit(X_train, y_train)
        print("Best hiperparámeters:", grid.best_params_)
        print(f"Best {self.scoring}: {grid.best_score_:.4f}")
        return grid.best_estimator_
