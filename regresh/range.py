import h5py
import numpy as np
import os

# Define the data folder
data_folder = 'testing_hdf'
range_array = []

# Get list of all HDF5 files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.endswith('.hdf5')]

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
            
            # Compute the standard deviation for each row
            range = np.ptp(first_image, axis=1)
            
            # Append the standard deviations to the output list
            range_array.extend(range)
    
    except Exception as e:
        # Print error message if file processing fails
        print(f'Failed to process file {filename}: {str(e)}')

# Convert the list of standard deviations to a numpy array
range_array = np.array(range_array).reshape(-1, 1)

# Save the standard deviations to a CSV file
np.savetxt('test_range_values.csv', range_array, delimiter=',')

# Display the shape of the final array
print(f'Final array shape: {range_array.shape}')

print('Processing complete.')