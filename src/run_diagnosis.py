import joblib
import pandas as pd
import numpy as np

model = joblib.load('rf_model.pkl')
symptoms = pd.read_csv('data/symptom_index.csv').columns.tolist()

input_text = input("Enter symptoms separated by commas: ").lower()
input_symptoms = [s.strip() for s in input_text.split(',')]

# Create symptom vector
vector = [1 if s in input_symptoms else 0 for s in symptoms]

prediction = model.predict([vector])
print("Predicted Disease:", prediction[0])
