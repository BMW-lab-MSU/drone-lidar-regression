import h5py
import numpy as np
import os

# Define the data folder
data_folder = 'testing_hdf'
median_array = []

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
            # Determine the range of indices based on filename prefix
            if filename.startswith('happy'):
                data_indices = range(121, 124)  # Indices 121-124 (Python indexing)
            elif filename.startswith('stan'):
                data_indices = range(118, 121)  # Indices 118-120 (Python indexing)
            else:
                continue
            
            # Assuming 'data/data' contains the images
            images = file['data']['data']
            
            # Extract the first image and select specified columns
            first_image = images[0]
            selected_rows = first_image[data_indices, :]
            
            # Compute the median for each row of the selected columns
            median = np.median(selected_rows, axis=1)
            
            # Append the median values to the output list
            median_array.extend(median)
    
    except Exception as e:
        # Print error message if file processing fails
        print(f'Failed to process file {filename}: {str(e)}')

# Convert the list of medians to a numpy array
median_array = np.array(median_array).reshape(-1, 1)

# Save the medians to a CSV file
np.savetxt('feature55.csv', median_array, delimiter=',')

# Display the shape of the final array
print(f'Final array shape: {median_array.shape}')

print('Processing complete.')
