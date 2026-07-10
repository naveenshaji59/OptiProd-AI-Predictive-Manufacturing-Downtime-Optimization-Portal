import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


print("Generating synthetic factory sensor logs...")
np.random.seed(42)
n_rows = 1000


temperature = np.random.normal(loc=70, scale=10, size=n_rows)      # Avg temp 70°C
vibration = np.random.normal(loc=4.0, scale=1.2, size=n_rows)     # Avg vibration 4mm/s
tool_wear_hours = np.random.uniform(0, 100, size=n_rows)          # Hours tool has been used


fail_condition = (temperature > 88) | ((vibration > 6.0) & (tool_wear_hours > 80))
failure = np.where(fail_condition, 1, 0) # 1 = Failed, 0 = Healthy


df = pd.DataFrame({
    'Temperature': temperature,
    'Vibration': vibration,
    'Tool_Wear_Hours': tool_wear_hours,
    'Failure': failure
})


X = df[['Temperature', 'Vibration', 'Tool_Wear_Hours']]
y = df['Failure']


print("Training the Predictive Maintenance Model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)


joblib.dump(model, 'predictive_model.pkl')
print("Success! Model saved as 'predictive_model.pkl'")