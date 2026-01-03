from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from threading import Thread
import subprocess
import json

app = Flask(__name__)
MODEL_CO_FILE = 'catboost_co_model.pkl'
MODEL_C6H6_FILE = 'catboost_c6h6_model.pkl'

# Confirmed 27 features from your training script
FEATURE_COLS = [
    'hour_sin', 'hour_cos', 
    'day_sin', 'day_cos',
    'CO_lag_1hr', 'CO_lag_24hr',
    'C6H6_lag_1hr', 'C6H6_lag_24hr',
    'CO_roll_mean_3hr', 'C6H6_roll_mean_3hr',
    'CO_roll_mean_24hr', 'CO_roll_std_24hr', 'CO_roll_max_24hr',
    'C6H6_roll_mean_24hr', 'C6H6_roll_std_24hr', 'C6H6_roll_max_24hr',
    'CO_diff_1hr', 'CO_diff_24hr',
    'C6H6_diff_1hr', 'C6H6_diff_24hr',
    'C6H6_diff_x_hour_sin', 'C6H6_diff_x_hour_cos',
    'CO_diff_x_hour_sin', 'CO_diff_x_hour_cos',
    'is_daytime',
    'is_morning_rush',
    'is_evening_rush'
]

def load_models():
    try:
        model_co = joblib.load(MODEL_CO_FILE)
        model_c6h6 = joblib.load(MODEL_C6H6_FILE)
        return model_co, model_c6h6
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

model_co, model_c6h6 = load_models()

# Global variable to store last prediction result
last_prediction = None

def featurize_single_input(data: dict) -> pd.DataFrame:
    # This server expects the CLIENT (your pipeline) to have pre-calculated all 27 features.
    try:
        df_input = pd.DataFrame([data])
        return df_input[FEATURE_COLS]
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}. The model expects all 27 features.")

@app.route('/predict', methods=['POST'])
def predict():
    if model_co is None or model_c6h6 is None:
        return jsonify({'error': 'Models not loaded. Check server logs.'}), 500
        
    try:
        json_data = request.json
        X_predict = featurize_single_input(json_data)

        co_pred = model_co.predict(X_predict)[0]
        c6h6_pred = model_c6h6.predict(X_predict)[0]
        
        response = {
            'CO(GT)_prediction': float(co_pred),
            'C6H6(GT)_prediction': float(c6h6_pred)
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/run-datapipe', methods=['POST'])
def run_datapipe_api():
    """API endpoint to trigger datapipe execution"""
    global last_prediction
    
    try:
        # Run datapipe as subprocess
        result = subprocess.run(['python', 'datapipe.py'], 
                              capture_output=True, 
                              text=True,
                              timeout=120)
        
        # Parse output to extract prediction results
        output = result.stdout
        
        # Extract predictions from last 3 lines containing the values
        lines = output.strip().split('\n')
        
        co_pred = None
        c6h6_pred = None
        pred_time = None
        
        # Search for the prediction values in output
        for i, line in enumerate(lines):
            if 'CO(GT):' in line:
                try:
                    co_pred = float(line.split(':')[1].strip())
                except:
                    pass
            if 'C6H6(GT):' in line:
                try:
                    c6h6_pred = float(line.split(':')[1].strip())
                except:
                    pass
            if 'Time:' in line and 'Prediction' in lines[i-1] if i > 0 else False:
                pred_time = line.split('Time:')[1].strip()
        
        # Fallback: if values not found, return error
        if co_pred is None or c6h6_pred is None:
            return jsonify({
                'success': False,
                'message': 'Failed to extract predictions from datapipe output'
            }), 400
        
        # Get current time + 1 hour for prediction time
        if not pred_time:
            pred_time = (datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).strftime('%Y-%m-%d %H:00')
        
        last_prediction = {
            'CO': co_pred,
            'C6H6': c6h6_pred,
            'time': pred_time
        }
        
        return jsonify({
            'success': True,
            'results': last_prediction
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'message': 'Datapipe execution timeout'
        }), 504
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


def run_datapipe_once():
    try:
        print("[DATAPIPE] Running datapipe automation...")
        print("="*60)
        
        # Run datapipe.py as subprocess to capture all output
        result = subprocess.run(['python', 'datapipe.py'], 
                              capture_output=True, 
                              text=True)
        
        # Print all output from datapipe
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[DATAPIPE STDERR]:", result.stderr)
        
        print("="*60)
        print(f"[DATAPIPE] Completed with exit code: {result.returncode}")
    except Exception as e:
        print(f"[DATAPIPE] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run datapipe once on server startup
    datapipe_thread = Thread(target=run_datapipe_once, daemon=True)
    datapipe_thread.start()
    
    print("[SERVER] Starting Flask server...\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)