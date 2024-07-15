import h5py
import numpy as np
import os

# Define the data folder
data_folder = 'testing_hdf'
iqr_array = []

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
            
            q1 = np.percentile(first_image, 25, axis=1)
            q3 = np.percentile(first_image, 75, axis=1)

            iqr = q3 - q1

            iqr_array.extend(iqr)
    
    except Exception as e:
        # Print error message if file processing fails
        print(f'Failed to process file {filename}: {str(e)}')

# Convert the list of MADs to a numpy array
iqr_array = np.array(iqr_array).reshape(-1, 1)

# Save the MADs to a CSV file
np.savetxt('test_interquartile_range_data.csv', iqr_array, delimiter=',')

# Display the shape of the final array
print(f'Final array shape: {iqr_array.shape}')

print('Processing complete.')
