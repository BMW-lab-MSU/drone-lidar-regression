import h5py
import numpy as np
import os

data_folder = 'train/train'
output_array = []

# Initialize RPM values array
rpm_values = []

# Iterate through HDF5 files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            with h5py.File(filepath, 'r') as file:
                # Extract RPM values from parameters/prop_frequency/front_right/avg
                rpms = file['parameters']['prop_frequency']['front_right']['avg'][:]
                
                # Append RPM values for this file to the rpm_values list
                rpm_values.append(rpms)
                
                # Determine the range of indices based on filename prefix
                if filename.startswith('happy'):
                    data_indices = range(120, 125)  # Indices 121-125 (Python indexing)
                elif filename.startswith('stan'):
                    data_indices = range(118, 128)  # Indices 119-127 (Python indexing)
                else:
                    continue
                
                # Initialize an array with zeros
                rpm_array = np.zeros(256)
                
                # Assign RPM values to the specified indices
                for index in data_indices:
                    rpm_array[index] = rpms[index % 32]  # Use modulo to cycle through rpms
                
                # Append the RPM values to the output array
                output_array.append(rpm_array)
        
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

# Stack all arrays in output_array into a single 1-dimensional array
if output_array:
    final_array = np.vstack(output_array).flatten()
    print(f'Final array shape: {final_array.shape}')
    
    # Optionally save to a CSV file
    np.savetxt('output_rpm_values.csv', final_array, delimiter=',')
else:
    print('No valid arrays found in output_array.')

print('Processing complete.')
