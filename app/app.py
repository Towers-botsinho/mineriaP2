import sys
import os
from flask import Flask, request, jsonify, render_template

# Add the 'src' directory to the Python path to import our modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from predict import LifeExpectancyPredictor

app = Flask(__name__)

# Load the model predictor once at start
try:
    predictor = LifeExpectancyPredictor()
    print("Model successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

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
