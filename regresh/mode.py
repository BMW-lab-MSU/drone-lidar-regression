import h5py
import numpy as np
import os
from collections import Counter

# Define the data folder
data_folder = 'testing_hdf'
mode_array = []

# Get list of all HDF5 files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.endswith('.hdf5')]

# Function to calculate the mode
def calculate_mode(data):
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return modes[0]  # Return the first mode if multiple modes are found

# Iterate through each file in the folder
for filename in file_list:
    filepath = os.path.join(data_folder, filename)
    
    try:
        # Print the current file being processed
        print(f'Processing file: {filename}')
        
        # Open the HDF5 file
        with h5py.File(filepath, 'r') as file:
            # Assuming 'data/data' contains the images
            images = file['data']['data']
            
            # Extract the first image
            first_image = images[0]
            
            # Compute the mode for each row
            for row in first_image:
                mode = calculate_mode(row)
                mode_array.append(mode)
    
    except Exception as e:
        # Print error message if file processing fails
        print(f'Failed to process file {filename}: {str(e)}')

# Convert the list of modes to a numpy array
mode_array = np.array(mode_array).reshape(-1, 1)

# Save the modes to a CSV file
np.savetxt('test_mode_values.csv', mode_array, delimiter=',')

# Display the shape of the final array
print(f'Final array shape: {mode_array.shape}')

print('Processing complete.')
