import numpy as np
import joblib
import csv

# Load the trained model and scaler
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new image data for prediction
new_image_data = np.loadtxt('tested_image_data.csv', delimiter=',')

# Standardize the new image data
new_image_data = scaler.transform(new_image_data)

# Predict RPM values for the new data
predicted_rpm = model.predict(new_image_data)

np.savetxt('prediction.csv', predicted_rpm, delimiter=',')

# Display the predicted RPM
print('Predicted RPM using the data:')
print(predicted_rpm)




