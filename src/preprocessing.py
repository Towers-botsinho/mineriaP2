import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Load data from a CSV file and perform initial cleaning.
    Fixes column names and drops rows with missing target variable.
    """
    df = pd.read_csv(filepath)
    
    # Clean column names (strip spaces, replace spaces with underscores, lower case)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # The target variable is 'life_expectancy'
    if 'life_expectancy' in df.columns:
        df = df.dropna(subset=['life_expectancy'])
        
    return df

def handle_missing_values(df):
    """
    Impute missing values. Numeric with median, Categorical with mode.
    """
    df = df.copy()
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            
    return df

def detect_and_treat_outliers(df, columns):
    """
    Treat outliers using the IQR method. 
    Caps values at the lower and upper bounds instead of dropping them.
    """
    df = df.copy()
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            
    return df

def validate_data_types(df):
    """
    Ensure correct data types.
    """
    df = df.copy()
    # explicitly convert country and status to string/category
    if 'country' in df.columns:
        df['country'] = df['country'].astype('category')
    if 'status' in df.columns:
        df['status'] = df['status'].astype('category')
        
    return df

def run_preprocessing_pipeline(filepath):
    """
    Run the full preprocessing pipeline.
    """
    df = load_and_clean_data(filepath)
    df = handle_missing_values(df)
    
    # We apply outlier treatment to specific numerical columns, excluding target and year
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'life_expectancy' in numeric_features:
        numeric_features.remove('life_expectancy')
    if 'year' in numeric_features:
        numeric_features.remove('year')
        
    df = detect_and_treat_outliers(df, numeric_features)
    df = validate_data_types(df)
    
    return df

if __name__ == "__main__":
    import os
    # For testing the pipeline
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'Life Expectancy Data.csv')
    
    if os.path.exists(data_path):
        df_clean = run_preprocessing_pipeline(data_path)
        print("Preprocessing successful. Shape:", df_clean.shape)
    else:
        print("Data file not found at:", data_path)
