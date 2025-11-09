from flask import Flask, request, jsonify, render_template
import pandas as pd
from capstone import xgb, x, encode_data

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  
    data = request.json
    new_data = pd.DataFrame([data])

    # Fill missing expected columns with default placeholder values
    for col in ['gender', 'enrolled_university', 'major_discipline', 'company_type', 'city']:
        if col not in new_data.columns:
            new_data[col] = 'Other'  # safe placeholder category

    # Encode using same training transformations
    new_data = encode_data(new_data, reference_cols=x.columns)

    # Convert numeric fields properly
    for col in ['city_development_index', 'training_hours']:
        if col in new_data.columns:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce').fillna(0)

    # Match column order
    new_data = new_data[x.columns].copy()

    # Predict using trained model
    pred = xgb.predict(new_data)[0]
    confidence = float(xgb.predict_proba(new_data)[0][int(pred)] * 100)
    result = "Likely to Stay (0)" if pred == 0 else "Likely to Leave (1)"

    return jsonify({
        'prediction': result,
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
