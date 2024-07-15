import h5py
import numpy as np
import os

data_folder = 'training_hdf'
output_array = []

for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            print(f'Processing file: {filename}')
            with h5py.File(filepath, 'r') as file:
                # Assuming 'data/data' contains the images
                images = file['data']['data']
                
                # Extract the first image
                first_image = images[0]
                output_array.append(first_image)
        
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

if output_array:
    final_array = np.vstack(output_array)
    print(f'Final array shape: {final_array.shape}')
    
    # Save the final array to a CSV file
    np.savetxt('trained_data.csv', final_array, delimiter=',')
else:
    print('No valid arrays found in output_array.')

print('Processing complete.')
