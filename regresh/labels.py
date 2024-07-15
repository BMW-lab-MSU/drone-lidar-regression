import h5py
import os
import numpy as np
import csv

# Define the directory containing the training data
data_dir = 'train/train'

# Initialize an empty list to store the RPM values
rpm_values = []

# Iterate over the files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".hdf5"):
        # Ensure we're only processing HDF5 files
        filepath = os.path.join(data_dir, filename)
        
        # Read RPM value from HDF5 file
        try:
            with h5py.File(filepath, 'r') as file:
                rpm_value = file['parameters/prop_frequency/front_right/avg'][()]
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
            continue
        
        # Check the filename and repeat the RPM value accordingly
        if filename.startswith('happy'):
            rpm_values.extend([rpm_value] * 5)
        elif filename.startswith('stan'):
            rpm_values.extend([rpm_value] * 10)

# Convert the list to a numpy array and reshape it into a column vector
rpm_array = np.array(rpm_values).reshape(-1, 1)

# Save the array to a CSV file
csv_file = 'rpm_values.csv'
np.savetxt(csv_file, rpm_array, delimiter=',')

print(f"CSV file '{csv_file}' successfully saved.")
