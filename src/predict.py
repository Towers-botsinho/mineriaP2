import os
import joblib
import pandas as pd
from features import feature_engineering_pipeline

class LifeExpectancyPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, 'model', 'model.pkl')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
            
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.preprocessor = artifacts['preprocessor']

    def predict(self, input_data):
        # Convert dictionary to DataFrame (assume single row or multiple rows)
        if isinstance(input_data, dict):
            # If it's a single record, make it a list of dicts
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            raise ValueError("input_data must be a dictionary or list of dictionaries")

        # Clean column names exactly as in load_and_clean_data
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # We assume the missing values are already handled or we impute them here 
        # (for a real API, we should require all fields or impute them)
        # To make it simple, we do basic imputation if missing
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0) # naive fill for API missing keys
        
        # Apply strict data types for categories if present
        if 'country' in df.columns:
            df['country'] = df['country'].astype('category')
        if 'status' in df.columns:
            df['status'] = df['status'].astype('category')
            
        # The feature_engineering_pipeline needs the exact same columns as training minus target.
        # It expects all columns present during training.
        X_processed, _ = feature_engineering_pipeline(df, is_training=False, preprocessor=self.preprocessor)
        
        predictions = self.model.predict(X_processed)
        return predictions

if __name__ == "__main__":
    # Test predict
    test_input = {
        "Country": "Afghanistan",
        "Year": 2015,
        "Status": "Developing",
        "Adult Mortality": 263.0,
        "infant deaths": 62,
        "Alcohol": 0.01,
        "percentage expenditure": 71.27962362,
        "Hepatitis B": 65.0,
        "Measles": 1154,
        "BMI": 19.1,
        "under-five deaths": 83,
        "Polio": 6.0,
        "Total expenditure": 8.16,
        "Diphtheria": 65.0,
        "HIV/AIDS": 0.1,
        "GDP": 584.25921,
        "Population": 33736494.0,
        " thinness  1-19 years": 17.2,
        " thinness 5-9 years": 17.3,
        "Income composition of resources": 0.479,
        "Schooling": 10.1
    }
    
    predictor = LifeExpectancyPredictor()
    pred = predictor.predict(test_input)
    print("Predicted Life Expectancy:", pred[0])
