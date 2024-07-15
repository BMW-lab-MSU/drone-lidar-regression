import h5py
import os
import numpy as np

# Define the directory containing the training data
data_dir = 'train/train'

# Initialize an empty list to store the stacked data
stacked_data = []

# Iterate over the files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".hdf5"):
        # Ensure we're only processing HDF5 files
        filepath = os.path.join(data_dir, filename)
        
        try:
            with h5py.File(filepath, 'r') as file:
                if filename.startswith('happy'):
                    # Extract rows 121 to 125 (indices 120 to 124)
                    data_chunk = file['data'][120:125, :]
                elif filename.startswith('stan'):
                    # Extract rows 119 to 128 (indices 118 to 127)
                    data_chunk = file['data'][118:128, :]
                else:
                    continue
                
                # Append to the stacked data
                stacked_data.append(data_chunk)
        
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue

# Stack all arrays in the list along the first axis (vertical stack)
stacked_array = np.vstack(stacked_data)

# Verify the shape of the stacked array
print(f"Final stacked array shape: {stacked_array.shape}")

