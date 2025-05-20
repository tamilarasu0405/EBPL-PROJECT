# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset (replace 'accidents.csv' with your actual dataset)
df = pd.read_csv('accidents.csv')

# Sample preprocessing: Drop nulls and convert categorical features
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['weather', 'road_condition', 'light_condition'])

# Features and label
X = df.drop('severity', axis=1)   # Features
y = df['severity']                # Labels (e.g., 0 = low, 1 = medium, 2 = high)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
