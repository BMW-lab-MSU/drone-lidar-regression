import h5py
import numpy as np
import os

# Define the data folder
data_folder = 'testing_hdf'
mean_absolute_deviation_array = []

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
            
            # Compute the mean for each row
            mean = np.mean(first_image, axis=1)
            
            # Compute the Mean Absolute Deviation for each row
            mad = np.mean(np.abs(first_image - mean[:, None]), axis=1)
            
            # Append the MAD to the output list
            mean_absolute_deviation_array.extend(mad)
    
    except Exception as e:
        # Print error message if file processing fails
        print(f'Failed to process file {filename}: {str(e)}')

# Convert the list of MADs to a numpy array
mean_absolute_deviation_array = np.array(mean_absolute_deviation_array).reshape(-1, 1)

# Save the MADs to a CSV file
np.savetxt('test_mean_absolute_deviation_data.csv', mean_absolute_deviation_array, delimiter=',')

# Display the shape of the final array
print(f'Final array shape: {mean_absolute_deviation_array.shape}')

print('Processing complete.')
