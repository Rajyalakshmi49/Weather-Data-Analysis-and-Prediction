"""
train_model.py
----------------
This script trains and evaluates three regression models to predict
TEMPERATURE from weather + date features, then saves the best-performing
model to disk using Joblib so the Streamlit app can load it later.

Steps performed:
    1. Load and clean the dataset (via weather_utils.py)
    2. Create date-based features
    3. Split the data CHRONOLOGICALLY (train = earlier dates, test = later
       dates) because this is time-series data. Shuffling would leak
       future information into training, which is not realistic.
    4. Train three models:
         - Linear Regression
         - Random Forest Regressor
         - Gradient Boosting Regressor
    5. Evaluate each model using MAE, MSE, RMSE and R^2
    6. Save the best model (highest R^2 on the test set) with Joblib
    7. Save a small "model_info" dictionary (metrics + feature list) so the
       Streamlit dashboard can display performance without retraining.

Run with:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from weather_utils import prepare_dataset, FEATURE_COLUMNS, TARGET_COLUMN

# ---------------------------------------------------------------------
# File paths (kept as constants so every script agrees on where things live)
# ---------------------------------------------------------------------
DATA_PATH = os.path.join("data", "weather_data.csv")
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.pkl")


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits a time-ordered DataFrame into train/test sets WITHOUT shuffling.
    The most recent `test_size` fraction of rows becomes the test set.
    This mimics a real-world scenario: train on the past, test on the future.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df


def evaluate_model(model, X_test, y_test) -> dict:
    """Computes MAE, MSE, RMSE and R^2 for a fitted model on test data."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    print("=" * 60)
    print("Weather Data Analysis and Temperature Prediction - Training")
    print("=" * 60)

    # 1 & 2. Load, clean and feature-engineer the dataset
    print("\n[1/5] Loading and cleaning dataset...")
    df = prepare_dataset(DATA_PATH)
    print(f"    Dataset ready with {len(df)} rows after cleaning.")

    # 3. Chronological train/test split
    print("\n[2/5] Splitting data chronologically (80% train / 20% test)...")
    train_df, test_df = chronological_split(df, test_size=0.2)
    print(f"    Train rows: {len(train_df)}  (dates {train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"    Test rows:  {len(test_df)}  (dates {test_df['date'].min().date()} to {test_df['date'].max().date()})")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # 4. Define the three models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    print("\n[3/5] Training models...")
    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"    Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print(
            f"        MAE={metrics['MAE']:.3f}  MSE={metrics['MSE']:.3f}  "
            f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}"
        )

    # 5. Select the best model (highest R^2 on the test set)
    print("\n[4/5] Selecting the best model based on R^2 score...")
    best_model_name = max(results, key=lambda name: results[name]["R2"])
    best_model = fitted_models[best_model_name]
    best_metrics = results[best_model_name]
    print(f"    Best model: {best_model_name}  (R2 = {best_metrics['R2']:.4f})")

    # 6 & 7. Save the best model and the model info summary
    print("\n[5/5] Saving best model and results with Joblib...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)

    model_info = {
        "best_model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "all_results": results,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": (str(train_df["date"].min().date()), str(train_df["date"].max().date())),
        "test_date_range": (str(test_df["date"].min().date()), str(test_df["date"].max().date())),
    }
    joblib.dump(model_info, MODEL_INFO_PATH)

    print(f"    Saved model to:      {BEST_MODEL_PATH}")
    print(f"    Saved model info to: {MODEL_INFO_PATH}")
    print("\nTraining complete! You can now run the Streamlit app with:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()