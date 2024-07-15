import h5py
import numpy as np
import os

data_folder = 'training_hdf'
output_array = []

# Iterate through HDF5 files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            with h5py.File(filepath, 'r') as file:
                # Extract RPM values from parameters/prop_frequency/front_right/avg
                rpms = file['parameters']['prop_frequency']['front_right']['avg'][:]
                
                # Determine the range of indices based on filename prefix
                if filename.startswith('happy'):
                    data_indices = range(121, 124)  # Indices 121-125 (Python indexing)
                elif filename.startswith('stan'):
                    data_indices = range(118, 121)  # Indices 119-127 (Python indexing)
                else:
                    continue
                
                # Initialize an array to store RPM values for each image
                image_rpm_values = rpms[:32]  # Assuming there are exactly 32 images
                
                # Loop through each image's RPM value
                for rpm_value in image_rpm_values:
                    # Initialize an array with zeros for each image
                    rpm_array = np.zeros(256)
                    
                    # Assign the current RPM value to the specified indices
                    for index in data_indices:
                        rpm_array[index] = rpm_value
                    
                    # Append the RPM values to the output array
                    output_array.append(rpm_array)
        
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

# Stack all arrays in output_array into a single 1-dimensional array
if output_array:
    final_array = np.vstack(output_array).flatten()
    print(f'Final array shape: {final_array.shape}')
    
    # Optionally save to a CSV file
    np.savetxt('rpm1_values.csv', final_array, delimiter=',')
else:
    print('No valid arrays found in output_array.')

print('Processing complete.')
