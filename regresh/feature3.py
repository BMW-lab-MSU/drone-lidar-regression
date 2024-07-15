import h5py
import numpy as np
import os

# Define the data folder
data_folder = 'testing_hdf'
mean_absolute_deviation_array = []

# Get list of all HDF5 files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.endswith('.hdf5')]

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
            
            # Extract the first image and select specified rows
            first_image = images[0]
            selected_rows = first_image[data_indices, :]
            
            # Compute the mean for each selected row
            mean = np.mean(selected_rows, axis=1)
            
            # Compute the Mean Absolute Deviation for each selected row
            mad = np.mean(np.abs(selected_rows - mean[:, None]), axis=1)
            
            # Append the MAD values to the output list
            mean_absolute_deviation_array.extend(mad)
    
    except Exception as e:
        # Print error message if file processing fails
        print(f'Failed to process file {filename}: {str(e)}')

# Convert the list of MADs to a numpy array
mean_absolute_deviation_array = np.array(mean_absolute_deviation_array).reshape(-1, 1)

# Save the MADs to a CSV file
np.savetxt('feature33.csv', mean_absolute_deviation_array, delimiter=',')

# Display the shape of the final array
print(f'Final array shape: {mean_absolute_deviation_array.shape}')

print('Processing complete.')
