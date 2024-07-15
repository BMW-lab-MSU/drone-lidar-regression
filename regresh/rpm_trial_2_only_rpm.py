import h5py
import numpy as np
import os

data_folder = 'testing_hdf'
output_array = []

# Iterate through HDF5 files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            with h5py.File(filepath, 'r') as file:
                # Extract RPM values from parameters/prop_frequency/front_right/avg
                rpms = file['parameters']['prop_frequency']['front_right']['avg'][:]
                
                # Determine the RPM value to use (take the first value)
                rpm_value = rpms[0]
                
                # Determine the range of indices based on filename prefix
                if filename.startswith('happy'):
                    data_indices = range(121, 124)  # Indices 121-125 (Python indexing)
                elif filename.startswith('stan'):
                    data_indices = range(118, 121)  # Indices 119-127 (Python indexing)
                else:
                    continue
                
                # Initialize an array with zeros for the RPM values
                selected_rpms = np.full((len(data_indices),), rpm_value)
                output_array.append(selected_rpms)


        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

if output_array:
    # Convert output_array to a numpy array
    final_array = np.vstack(output_array)

    # Reshape to a column vector
    final_array = final_array.reshape(-1, 1)

    
    print(f'Final array shape: {final_array.shape}')

    # Save to a CSV file
    np.savetxt('test_rpm1_only_values.csv', final_array, delimiter=',')

else:
    print('No valid arrays in the output array')

print('Processing complete.')
