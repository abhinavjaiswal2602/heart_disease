from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_DIR = 'models'
all_models = joblib.load(os.path.join(MODEL_DIR, 'all_models.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Age band: <40=0, 40-55=1, 55-70=2, >70=3
    df['age_band'] = pd.cut(
        df['age'],
        bins=[0, 40, 55, 70, 120],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)

    # Composite risk score
    df['risk_score'] = (
        df['exang'].astype(int) + 
        df['fbs'].astype(int) + 
        (df['oldpeak'] > 1.5).astype(int) + 
        (df['ca'] > 0).astype(int)
    )

    # Cast categorical columns to float to match trained feature names (e.g. cp_2.0)
    for cat_col in ['cp', 'thal', 'slope', 'restecg']:
        df[cat_col] = df[cat_col].astype(float)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['cp'], prefix='cp')
    df = pd.get_dummies(df, columns=['thal'], prefix='thal')
    df = pd.get_dummies(df, columns=['slope'], prefix='slope')
    df = pd.get_dummies(df, columns=['restecg'], prefix='restecg')
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    # Ensure order
    df = df[feature_columns]
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('plots', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure correct data types
        for col in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Feature Engineering
        df_engineered = feature_engineering(df)
        
        # Scaling
        X_scaled = scaler.transform(df_engineered)
        
        # Predictions from all models
        predictions = {}
        for name, model in all_models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
            predictions[name] = {
                'prediction': int(pred),
                'probability': float(prob)
            }
            
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
