import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ✅ Load local dataset
data = pd.read_csv("weatherAUS.csv")

# ✅ Select features and clean data
data = data[['Humidity3pm', 'Pressure9am', 'Temp3pm', 'RainTomorrow']]
data.dropna(inplace=True)

# ✅ Encode target column
le = LabelEncoder()
data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

# ✅ Features and label
X = data[['Humidity3pm', 'Pressure9am', 'Temp3pm']]
y = data['RainTomorrow']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", round(accuracy * 100, 2), "%")

# ✅ Predict tomorrow's rain using new data
sample = [[55, 1012, 23]]  # Humidity3pm, Pressure9am, Temp3pm
prediction = model.predict(sample)
print("🌧️ Will it rain tomorrow?", "Yes" if prediction[0] == 1 else "No")
