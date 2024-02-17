from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
try:
    with open('diabetes_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print("Error: Model files not found. Make sure you have the necessary model files in the correct location.")
    clf = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or scaler is None:
        return jsonify("Error: Model not loaded. Please check server logs for details.")

    data = request.get_json(force=True)
    pregnancies = data['pregnancies']
    glucose = data['glucose']
    blood_pressure = data['blood_pressure']
    skin_thickness = data['skin_thickness']
    insulin = data['insulin']
    bmi = data['bmi']
    diabetes_pedigree_function = data['diabetes_pedigree_function']
    age = data['age']

    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    user_input_scaled = scaler.transform(user_input)
    
    try:
        prediction = clf.predict(user_input_scaled)

        if prediction == 0:
            result = "The model predicts that the person does not have diabetes."
        else:
            result = "The model predicts that the person has diabetes."

        return jsonify(result)
    except Exception as e:
        return jsonify("Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
