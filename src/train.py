import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import run_preprocessing_pipeline
from features import feature_engineering_pipeline

def evaluate_model(y_true, y_pred, split_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- Metrics for {split_name} Split ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}\n")
    return rmse, mae, r2

def train_and_evaluate(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    split_name = f"{int((1 - test_size) * 100)}/{int(test_size * 100)}"
    metrics = evaluate_model(y_test, y_pred, split_name)
    
    return model, metrics

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'Life Expectancy Data.csv')
    model_dir = os.path.join(BASE_DIR, 'model')
    
    print("Loading and preprocessing data...")
    df_clean = run_preprocessing_pipeline(data_path)
    
    print("Performing feature engineering...")
    X_processed, y, preprocessor = feature_engineering_pipeline(df_clean, is_training=True)
    
    print("Training models with different splits...")
    # 70/30 split
    model_7030, metrics_7030 = train_and_evaluate(X_processed, y, test_size=0.3)
    
    # 80/20 split
    model_8020, metrics_8020 = train_and_evaluate(X_processed, y, test_size=0.2)
    
    # We will serialize the Pipeline containing both the preprocessor and the best model (e.g. 80/20 model)
    # Actually, feature engineering has some pandas custom logic (like log transform and derived features) 
    # before applying the Scikit-Learn preprocessor. We must save the sklearn preprocessor and the regression model.
    # The API will just call the same python feature_engineering_pipeline with is_training=False.
    
    model_path = os.path.join(model_dir, 'model.pkl')
    
    # Save a dictionary with the necessary artifacts
    artifacts = {
        'model': model_8020,
        'preprocessor': preprocessor
    }
    
    joblib.dump(artifacts, model_path)
    print(f"Model and preprocessor successfully serialized to {model_path}")

if __name__ == "__main__":
    main()
