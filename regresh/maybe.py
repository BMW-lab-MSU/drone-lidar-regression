import h5py
import numpy as np
import os
import psutil


# Set the path to the folder containing HDF5 files
data_folder = 'training_hdf'
output_array = []  # List to collect standard deviations

# Process each HDF5 file in the folder
for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):  
        filepath = os.path.join(data_folder, filename)
        
        try:
            print(f'Processing file: {filename}')  

            with h5py.File(filepath, 'r') as file:
                images = file['data']['data'][:]
                
                for row in range(1, 257):
                    stddev_per_row = np.std(images, axis=0)
                    output_array.append(stddev_per_row)

                print("Current size of one dimentional array: ", len(output_array))
                
        

        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

# Convert list to NumPy array
stddev_array = np.array(output_array)

# Reshape to (3874816, 1)
stddev_array = stddev_array.reshape(-1, 1)

# Print information about the final array
print(f'Final array shape: {stddev_array.shape}')  # Should be (3874816, 1)

# Save the 1D array to a CSV file
np.savetxt('stddev_data.csv', stddev_array, delimiter=',', header='Standard Deviation', comments='')

print('Processing complete.')
