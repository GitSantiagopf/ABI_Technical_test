# main.py

import warnings
warnings.filterwarnings('ignore')

from data_pipeline.data_loader import DataLoader
from data_pipeline.data_explorer import DataExplorer
from data_pipeline.data_cleaner import DataCleaner
from data_pipeline.feature_engineer import FeatureEngineer
from data_pipeline.data_transformer import DataTransformer


from modeling.model_trainer import ModelTrainer
from modeling.model_evaluator import ModelEvaluator
from modeling.model_calibrator import ModelCalibrator
from modeling.predictor import Predictor
from modeling.clustering import Clusterer
from modeling.threshold_analyzer import ThresholdAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb

if __name__ == "__main__":
    #CREDIT SCORING

    # load data
    loader = DataLoader('data/data_raw/cs-training.csv')
    df = loader.load_data()
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # cleaning and imputation
    cleaner = DataCleaner()
    df = cleaner.dropna_in_column(df, 'MonthlyIncome')
    df = cleaner.fillna_with_value(df, 'NumberOfDependents', 0)
    
    # feature Engineering 
    engineer = FeatureEngineer()
    df = engineer.log_transform(df, ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines'])
    df = engineer.add_total_morosidad(df,
                                      col_30='NumberOfTime30-59DaysPastDueNotWorse',
                                      col_60='NumberOfTime60-89DaysPastDueNotWorse',
                                      col_90='NumberOfTimes90DaysLate')
    df = engineer.add_income_to_debt(df,
                                     log_income_col='log_MonthlyIncome',
                                     log_debt_col='log_DebtRatio')
    
    # data exploratory
    explorer = DataExplorer(df)
    explorer.info_data()
    variables_to_plot = ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'age']
    for var in variables_to_plot:
        explorer.plot_distribution(var, apply_log=False, save=True)
        explorer.plot_distribution(var, apply_log=True, save=True)
    corr_columns = ['SeriousDlqin2yrs', 'log_MonthlyIncome', 'log_DebtRatio',
                    'log_RevolvingUtilizationOfUnsecuredLines', 'age']
    explorer.correlation_heatmap(corr_columns, save=True)
    
    # train test split and data transform
    features_cs = [
        'log_MonthlyIncome', 'log_DebtRatio', 'log_RevolvingUtilizationOfUnsecuredLines',
        'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans',
        'NumberRealEstateLoansOrLines', 'NumberOfDependents', 'Total_Morosidad',
        'Income_to_Debt'
    ]
    target = 'SeriousDlqin2yrs'
    X_cs = df[features_cs]
    y_cs = df[target]
    X_train_cs, X_test_cs, y_train_cs, y_test_cs = train_test_split(
        X_cs, y_cs, test_size=0.2, random_state=42, stratify=y_cs
    )
    
    transformer = DataTransformer()
    X_train_cs_scaled = transformer.fit_transform(X_train_cs, features_cs)
    X_test_cs_scaled = transformer.transform(X_test_cs, features_cs)
    
    # train and hiperparameter tuning
    lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt_param_grid = {'max_depth': [None, 5, 10, 20],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4]}
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_param_grid = {'n_estimators': [50, 100],
                     'max_depth': [None, 5],
                     'min_samples_split': [2, 5]}
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_param_grid = {'n_estimators': [50, 100],
                      'max_depth': [3, 5],
                      'learning_rate': [0.01, 0.1]}
    ada = AdaBoostClassifier(random_state=42)
    ada_param_grid = {'n_estimators': [50, 100],
                      'learning_rate': [0.01, 0.1]}
    
    trainer = ModelTrainer(scoring='recall')
    best_lr = trainer.run_grid_search(lr, lr_param_grid, X_train_cs_scaled, y_train_cs)
    best_dt = trainer.run_grid_search(dt, dt_param_grid, X_train_cs_scaled, y_train_cs)
    best_rf = trainer.run_grid_search(rf, rf_param_grid, X_train_cs_scaled, y_train_cs)
    best_xgb = trainer.run_grid_search(xgb_clf, xgb_param_grid, X_train_cs_scaled, y_train_cs)
    best_ada = trainer.run_grid_search(ada, ada_param_grid, X_train_cs_scaled, y_train_cs)
    
    models_best_cs = {
        "Logistic Regression": best_lr,
        "Decision Tree": best_dt,
        "Random Forest": best_rf,
        "XGBoost": best_xgb,
        "AdaBoost": best_ada
    }
    
    # credit score model evaluate
    evaluator = ModelEvaluator()
    results_df_cs = evaluator.evaluate_models(models_best_cs, X_test_cs_scaled, y_test_cs, positive_label=1)
    evaluator.plot_confusion_matrices(models_best_cs, X_test_cs_scaled, y_test_cs, save=True)
    print("\Metrics summary risk credit score:")
    print(results_df_cs)
    print("\nRecall is principal metric")
    
    # probability calibration
    best_model_name = "Decision Tree"
    best_model_cs = models_best_cs[best_model_name]
    calibrator = ModelCalibrator(best_model_cs, method='sigmoid')
    calibrated_model = calibrator.calibrate(X_train_cs_scaled, y_train_cs)
    
    # plot curves
    evaluator.plot_calibration_curve(calibrated_model, X_train_cs_scaled, y_train_cs,
                                     model_name="DecisionTree_Calibrated", save=True)
    evaluator.plot_roc_curve(calibrated_model, X_test_cs_scaled, y_test_cs,
                             model_name="DecisionTree_Calibrated", save=True)
    
    # save predictions
    predictor = Predictor(calibrated_model, transformer.scaler, features_cs)
    predictor.predict_csv('data/data_raw/cs-test.csv', 'data/data_results/predictions_cs_test_calibrado.csv')

    # analysis for non-calibrated model
    best_model_name = "Decision Tree"
    best_model_uncalibrated = models_best_cs[best_model_name]  # Assume models_best_cs is defined
    analyzer_non_cal = ThresholdAnalyzer(best_model_uncalibrated, X_test_cs_scaled, y_test_cs)
    metrics_non_cal = analyzer_non_cal.analyze()
    analyzer_non_cal.plot_metrics(save=True)
    conclusion_non_cal = analyzer_non_cal.best_threshold_conclusion()

    # analysis for calibrated model
    analyzer_cal = ThresholdAnalyzer(calibrated_model, X_test_cs_scaled, y_test_cs)
    metrics_cal = analyzer_cal.analyze()
    analyzer_cal.plot_metrics(save=True)
    conclusion_cal = analyzer_cal.best_threshold_conclusion()

    # plot and save confusion matrices for various thresholds for the calibrated model
    analyzer_cal.plot_confusion_matrices_by_threshold(calibrated_model, X_test_cs_scaled, y_test_cs, save=True)
    
    
    # CLUSTERING
    # Load data
    df_cluster = loader.load_data()
    if 'Unnamed: 0' in df_cluster.columns:
        df_cluster.drop('Unnamed: 0', axis=1, inplace=True)
    df_cluster = df_cluster.dropna(subset=['MonthlyIncome'])
    df_cluster['NumberOfDependents'] = df_cluster['NumberOfDependents'].fillna(0)
    
    # feature engineering
    df_cluster = engineer.add_total_morosidad(df_cluster,
                                              col_30='NumberOfTime30-59DaysPastDueNotWorse',
                                              col_60='NumberOfTime60-89DaysPastDueNotWorse',
                                              col_90='NumberOfTimes90DaysLate')
    
    # features
    cluster_features = [
        'MonthlyIncome',
        'DebtRatio',
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberRealEstateLoansOrLines',
        'NumberOfDependents',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'Total_Morosidad'
    ]
    
    # clustering execute
    clusterer = Clusterer(df_cluster, cluster_features)
    X_scaled_cluster = clusterer.preprocess()
    clusterer.run_kmeans(n_clusters=3)
    clusterer.run_dbscan(eps=0.5, min_samples=10)
    clusterer.plot_metric_curves()
    clusterer.plot_boxplots()
    clusterer.plot_pca()
    clusterer.save_results("client_segmentation.csv")
