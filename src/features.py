import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def apply_log_transform(df, columns):
    """
    Apply logarithmic transformation to highly skewed features.
    Adding a small constant to avoid log(0).
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # check for negative values
            min_val = df[col].min()
            if min_val < 0:
                df[f'{col}_log'] = np.log1p(df[col] - min_val) # shift to positive
            else:
                df[f'{col}_log'] = np.log1p(df[col])
            # drop the original to avoid high multicollinearity if we want, or keep it. 
            # We'll replace the original column with log transformed one for simplicity.
            df.drop(columns=[col], inplace=True)
            df.rename(columns={f'{col}_log': col}, inplace=True)
    return df

def create_derived_features(df):
    """
    Create new derived variables based on domain knowledge or combinations.
    """
    df = df.copy()
    # Example 1: infant deaths ratio = infant deaths / under-five deaths
    # Prevent division by zero explosion when under_five_deaths is 0 (e.g. from missing imputation)
    if 'infant_deaths' in df.columns and 'under_five_deaths' in df.columns:
        # If under_five_deaths is exactly 0, ratio should be 0 to avoid massive numbers
        df['infant_to_under_five_ratio'] = np.where(
            df['under_five_deaths'] <= 0,
            0.0,
            df['infant_deaths'] / df['under_five_deaths']
        )
        
    # Example 2: total_health_expenditure_approx = gdp * (percentage_expenditure/100)
    # This is a rough proxy if GDP is per capita
    if 'gdp' in df.columns and 'percentage_expenditure' in df.columns:
        df['health_expenditure_approx'] = df['gdp'] * (df['percentage_expenditure'] / 100.0)
        
    return df

def feature_engineering_pipeline(df, is_training=True, preprocessor=None):
    """
    Perform encoding, scaling, transformations and feature creation.
    If is_training is True, fits a new preprocessor.
    If is_training is False, transform using the provided preprocessor.
    """
    df = df.copy()
    
    # 1. Transformations
    skewed_columns = ['gdp', 'population', 'percentage_expenditure']
    df = apply_log_transform(df, skewed_columns)
    
    # 2. Derived features
    df = create_derived_features(df)
    
    # Identify feature types
    target = 'life_expectancy'
    
    # We drop 'country' and 'year' typically to avoid massive dimensionality and overfitting in a simple baseline,
    # but let's keep 'year' as numerical and drop 'country'.
    cols_to_drop = ['country']
    if target in df.columns:
        cols_to_drop.append(target)
        
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target] if target in df.columns else None
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    if is_training:
        # Create preprocessor: StandardScaler for numeric, OneHotEncoder for categorical
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names back
        # numerical remain the same
        num_cols = numeric_features
        # categorical are extracted from the encoder
        if categorical_features:
            cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            feature_names = list(num_cols) + list(cat_cols)
        else:
            feature_names = num_cols
            
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        return X_processed_df, y, preprocessor
        
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided if is_training is False")
        
        X_processed = preprocessor.transform(X)
        
        # Get feature names
        num_cols = numeric_features
        if categorical_features:
            # Handle unknown categories gracefully using the fitted encoder's feature names (might be complex if categories were ignored)
            # but usually get_feature_names_out matches the shape
            cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            feature_names = list(num_cols) + list(cat_cols)
        else:
            feature_names = num_cols
            
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        return X_processed_df, y

if __name__ == "__main__":
    from preprocessing import run_preprocessing_pipeline
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'Life Expectancy Data.csv')
    
    if os.path.exists(data_path):
        df_clean = run_preprocessing_pipeline(data_path)
        X_train, y_train, preprocessor = feature_engineering_pipeline(df_clean, is_training=True)
        print("Feature Engineering successful.")
        print("Processed features shape:", X_train.shape)
        print("Sample columns:", X_train.columns[:5].tolist())
