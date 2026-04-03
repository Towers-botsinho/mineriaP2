import sys
import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Add the 'src' directory to the Python path to import our modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from predict import LifeExpectancyPredictor

app = Flask(__name__)

# Load dataset for frontend dynamic population
global_df = None
try:
    data_path = os.path.join(BASE_DIR, 'data', 'Life Expectancy Data.csv')
    df = pd.read_csv(data_path)
    # Replace NaN with None so it serializes properly to JSON
    global_df = df.replace({np.nan: None})
    print("Dataset loaded successfully for dynamic UI.")
except Exception as e:
    print(f"Error loading CSV dataset: {e}")

# Load the model predictor once at start
try:
    predictor = LifeExpectancyPredictor()
    print("Model successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

@app.route('/api/countries', methods=['GET'])
def get_countries():
    if global_df is None:
        return jsonify([])
    countries = sorted([c for c in global_df['Country'].unique() if c])
    return jsonify(countries)

@app.route('/api/years', methods=['GET'])
def get_years():
    if global_df is None:
        return jsonify([])
    country = request.args.get('country')
    if not country:
        return jsonify([])
    country_data = global_df[global_df['Country'] == country]
    years = sorted([int(y) for y in country_data['Year'].unique() if pd.notnull(y)], reverse=True)
    return jsonify(years)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    if global_df is None:
        return jsonify({})
    country = request.args.get('country')
    year = request.args.get('year', type=int)
    if not country or not year:
        return jsonify({})
    
    row = global_df[(global_df['Country'] == country) & (global_df['Year'] == year)]
    if row.empty:
        return jsonify({})
        
    stats = row.iloc[0].to_dict()
    # Remove Country and Year from stats to keep payload clean
    stats.pop('Country', None)
    stats.pop('Year', None)
    # Remove any None values
    stats = {k: v for k, v in stats.items() if v is not None}
    return jsonify(stats)

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Model object not initialized."}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided in the request"}), 400
            
        # Optional: Validate data
        if not isinstance(data, dict):
            return jsonify({"error": "Data must be a JSON object"}), 400
            
        prediction = predictor.predict(data)
        
        return jsonify({
            "prediction": float(prediction[0])
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
