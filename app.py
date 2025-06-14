from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load and train model
data = pd.read_csv("weatherAUS.csv")
data = data[['Humidity3pm', 'Pressure9am', 'Temp3pm', 'RainTomorrow']]
data.dropna(inplace=True)
le = LabelEncoder()
data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
X = data[['Humidity3pm', 'Pressure9am', 'Temp3pm']]
y = data['RainTomorrow']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    sample = [[
        float(input_data['Humidity3pm']),
        float(input_data['Pressure9am']),
        float(input_data['Temp3pm'])
    ]]
    prediction = model.predict(sample)
    result = "Yes" if prediction[0] == 1 else "No"
    return jsonify({"rain_tomorrow": result})

if __name__ == '__main__':
    app.run(debug=True)
