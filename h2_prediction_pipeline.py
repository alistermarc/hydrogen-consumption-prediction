"""
This script runs an end-to-end machine learning pipeline to predict hydrogen consumption.
The pipeline includes:
1. Data loading and preprocessing.
2. Exploratory Data Analysis (EDA), with plots saved to an output directory.
3. Outlier removal using Kernel Density Estimation (KDE).
4. Feature selection using Recursive Feature Elimination with Cross-Validation (RFECV).
5. Hyperparameter tuning for RandomForest, ExtraTrees, and LightGBM models using Optuna.
6. Training of the best model with the best hyperparameters.
7. Evaluation of the final model.
8. Saving the final trained model to a file.
"""

import os
import time
import warnings
from datetime import datetime

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = "..." # confidential
OUTPUT_DIR = "output"
MODEL_FILENAME = os.path.join(OUTPUT_DIR, "final_hydrogen_consumption_model.joblib")
N_TRIALS_OPTUNA = 50 

# --- Function Definitions ---

def load_and_preprocess_data(file_path):
    """Loads data from an Excel file and performs initial preprocessing and feature engineering."""
    print("1. Loading and preprocessing data...")
    df = pd.read_excel(file_path, sheet_name="HDT-1")

    # Convert 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Feature Engineering from dates
    df['Hour'] = df['Timestamp'].dt.hour
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month

    # Calculate Catalyst Age
    reference_date = datetime(2015, 1, 1)
    df['Catalyst_Age'] = (df['Timestamp'] - reference_date).dt.days

    # Select features and target
    features = df[['Feed_1', 'Feed_2', 'Feed_3', 'Feed_4', 'Feed_5', 'Feed_6', 'Sulfur_Product', 'Inlet_Temp', 'Catalyst_Age']]
    target = df["H2_Consumption"]

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df, features, target

def perform_eda(df, features, target, output_dir):
    """Performs exploratory data analysis and saves plots to the output directory."""
    print("2. Performing Exploratory Data Analysis (EDA)...")
    
    # Plot H2 Consumption over time
    plt.figure(figsize=(15, 7))
    plt.plot(df['Timestamp'], df['H2_Consumption'])
    plt.title('H2 Consumption Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('H2 Consumption')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "h2_consumption_over_time.png"))
    plt.close()

    # Correlation Matrix Heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = features.join(target).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    print("EDA plots saved to output directory.")

def remove_outliers(features, target, contamination_level=0.03):
    """Removes outliers from the dataset using Kernel Density Estimation."""
    print("3. Removing outliers...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kde = KernelDensity(kernel='gaussian').fit(features_scaled)
    scores_kde = kde.score_samples(features_scaled)
    threshold = np.quantile(scores_kde, contamination_level)

    outlier_mask = scores_kde > threshold
    features_filtered = features[outlier_mask]
    target_filtered = target[outlier_mask]

    print(f"Original number of samples: {len(features)}")
    print(f"Number of samples after outlier removal: {len(features_filtered)}")
    return features_filtered, target_filtered

def select_features(estimator, X, y, model_name, output_dir):
    """Selects the best features using RFECV and saves the results plot."""
    print(f"4. Selecting features with RFECV using {model_name}...")
    rfecv = RFECV(estimator=estimator, step=1, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rfecv.fit(X, y)

    print(f"Optimal number of features for {model_name}: {rfecv.n_features_}")

    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (RMSE)")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), -rfecv.cv_results_['mean_test_score'])
    plt.title(f'RFECV with {model_name}')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"rfecv_{model_name}.png"))
    plt.close()

    best_features = X.columns[rfecv.support_].tolist()
    print(f"Best features for {model_name}: {best_features}")
    return best_features

def tune_hyperparameters(model_name, X_train, y_train, n_trials):
    """Tunes hyperparameters for a given model using Optuna."""
    
    def objective(trial):
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 50, 200),
                'max_depth': trial.suggest_int("max_depth", 10, 30),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestRegressor(**params, random_state=0, n_jobs=-1)
        elif model_name == 'ExtraTrees':
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 50, 200),
                'max_depth': trial.suggest_int("max_depth", 10, 30),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
            }
            model = ExtraTreesRegressor(**params, random_state=0, n_jobs=-1)
        elif model_name == 'LGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
            }
            model = lgb.LGBMRegressor(**params, random_state=0, n_jobs=-1)
        else:
            raise ValueError("Unsupported model_name")

        score = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
        return -score.mean()

    print(f"--- Tuning {model_name} ---")
    start_time = time.time()
    sampler = TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    tuning_time = time.time() - start_time

    print(f"Tuning Time for {model_name}: {tuning_time:.2f}s")
    print(f"Best RMSE for {model_name}: {study.best_value:.4f}")
    print(f"Best Params for {model_name}: {study.best_params}")
    
    return study

def train_and_evaluate_final_model(model_name, best_params, X_train, y_train, X_test, y_test):
    """Trains the final model with the best hyperparameters and evaluates it on the test set."""
    print(f"6. Training and evaluating final model ({model_name})...")
    if model_name == 'RandomForest':
        final_model = RandomForestRegressor(**best_params, random_state=0, n_jobs=-1)
    elif model_name == 'ExtraTrees':
        final_model = ExtraTreesRegressor(**best_params, random_state=0, n_jobs=-1)
    else:  # LGBM
        final_model = lgb.LGBMRegressor(**best_params, random_state=0, n_jobs=-1)

    final_model.fit(X_train, y_train)
    test_preds = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    
    return final_model, test_rmse, test_r2

def save_model(model, file_path):
    """Saves the trained model to a file."""
    print("7. Saving final model...")
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def main():
    """Main function to run the entire ML pipeline."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load and Preprocess Data
    df, features, target = load_and_preprocess_data(DATA_FILE)

    # 2. EDA
    perform_eda(df, features, target, OUTPUT_DIR)

    # 3. Outlier Removal
    features_clean, target_clean = remove_outliers(features, target)

    # 4. Feature Selection
    # Using a subset of data for faster feature selection
    X_train_fs, _, y_train_fs, _ = train_test_split(features_clean, target_clean, test_size=0.7, random_state=0)
    rf_estimator = RandomForestRegressor(n_estimators=50, random_state=0)
    best_features = select_features(rf_estimator, X_train_fs, y_train_fs, "RandomForest", OUTPUT_DIR)
    
    # 5. Prepare data for modeling
    X = features_clean[best_features]
    y = target_clean
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 6. Hyperparameter Tuning
    print("\n5. Tuning hyperparameters...")
    models_to_tune = ['RandomForest', 'ExtraTrees', 'LGBM']
    studies = {}
    for model_name in models_to_tune:
        studies[model_name] = tune_hyperparameters(model_name, X_train, y_train, n_trials=N_TRIALS_OPTUNA)

    # 7. Final Model Selection and Training
    best_model_name = min(studies, key=lambda k: studies[k].best_value)
    best_study = studies[best_model_name]
    
    print(f"\n--- Best Model Selected: {best_model_name} ---")
    print(f"Best CV RMSE: {best_study.best_value:.4f}")
    print(f"Best Hyperparameters: {best_study.best_params}")

    final_model, test_rmse, test_r2 = train_and_evaluate_final_model(
        best_model_name, 
        best_study.best_params, 
        X_train, y_train, X_test, y_test
    )
    
    print(f"\nFinal Model Test RMSE: {test_rmse:.4f}")
    print(f"Final Model Test R2 Score: {test_r2:.4f}")

    # 8. Save Model
    save_model(final_model, MODEL_FILENAME)
    
    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
