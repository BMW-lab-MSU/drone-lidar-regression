import h5py
import numpy as np
import os

data_folder = 'testing_hdf'
output_array = []

for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            print(f'Processing file: {filename}')
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
                output_array.append(selected_rows)
        
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

if output_array:
    final_array = np.vstack(output_array)
    print(f'Final array shape: {final_array.shape}')
    
    # Save the final array to a CSV file
    np.savetxt('tested_stacked_rpm_data.csv', final_array, delimiter=',')
else:
    print('No valid arrays found in output_array.')

print('Processing complete.')
