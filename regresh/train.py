import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler

# Load the image data and RPM values from CSV files
image_data = np.loadtxt('stacked_image_data.csv', delimiter=',')
rpm_values = np.loadtxt('rpm1_values.csv', delimiter=',')


rpm_values = rpm_values.reshape(-1, 1)

# Check the shapes of the data
print(f'Image data shape: {image_data.shape}')
print(f'RPM values shape: {rpm_values.shape}')

# Ensure the number of samples match
assert image_data.shape[0] == rpm_values.shape[0], "Mismatch in the number of samples between image data and RPM values"

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(image_data)
y_train = rpm_values.ravel()

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'trained_model.pkl'
joblib.dump(model, model_filename)
print(f'Trained model saved as {model_filename}')

# Optionally, save the scaler to use for new data
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f'Scaler saved as {scaler_filename}')

# Optionally, display model information
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')