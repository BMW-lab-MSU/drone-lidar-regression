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
                blades = file['parameters']['n_blades'][()]
                
                # Determine the RPM value to use (take the first value)
                num_blades = blades
                
                # Determine the range of indices based on filename prefix
                if filename.startswith('happy'):
                    data_indices = range(121, 124)  # Indices 121-125 (Python indexing)
                elif filename.startswith('stan'):
                    data_indices = range(118, 121)  # Indices 119-127 (Python indexing)
                else:
                    continue
                
                # Initialize an array with zeros for the RPM values
                blade_array = np.zeros(256)
                
                # Assign the RPM value to the specified indices
                for index in data_indices:
                    blade_array[index] = num_blades
                
                # Append the RPM values to the output array
                output_array.extend(blade_array)
        
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

# Convert output_array to a numpy array
final_array = np.array(output_array)

# Reshape to a column vector
final_array = final_array.reshape(-1, 1)

# Save to a CSV file
np.savetxt('numblades_labels.csv', final_array, delimiter=',')

print(f'Final array shape: {final_array.shape}')
print('Processing complete.')
