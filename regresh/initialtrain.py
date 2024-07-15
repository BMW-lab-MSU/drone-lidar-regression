import os
import h5py
import numpy as np

folder_path = 'train/train'  # Adjust folder path as needed

# Get a list of all HDF5 files in the specified folder
files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]

# Initialize arrays to store extracted data
happy_data = []
stan_data = []

# Loop through each HDF5 file
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    print(f'Processing file: {file_path}')
    
    try:
        # Determine extraction rows based on filename prefix
        if file_name.startswith('happymodel'):
            rows_extracted = range(121, 126)  # Extract rows 121 to 125 for 'happymodel' files
        elif file_name.startswith('stan'):
            rows_extracted = range(119, 129)  # Extract rows 119 to 128 for 'stan' files
        else:
            print(f'Skipping file {file_name} with unknown prefix')
            continue  # Skip to the next file
        
        # Read data from HDF5 file
        with h5py.File(file_path, 'r') as f:
            dataMat = np.transpose(f['/data/data'], (2, 1, 0))  # Read and permute data
            timestampsMat = np.transpose(f['/data/timestamps'], (1, 0))  # Read and permute timestamps
            captureTime = f['/data/capture_time'][()]  # Read capture time
        
        # Verify the dimensions of dataMat
        num_captures, num_samples, num_segments = dataMat.shape
        print(f'Data dimensions (captures x samples x segments): {num_captures} x {num_samples} x {num_segments}')
        
        # Verify that the data has enough rows for extraction
        if num_samples < max(rows_extracted):
            print(f'Skipping file {file_name}: not enough rows for extraction')
            continue  # Skip to the next file
        
        # Extract specified rows
        extracted_rows = dataMat[:, rows_extracted, :]
        print(f'Extracted rows dimensions: {extracted_rows.shape}')
        
        # Append to respective data arrays based on prefix
        if file_name.startswith('happymodel'):
            happy_data.append(extracted_rows.reshape(-1, extracted_rows.shape[1] * extracted_rows.shape[2]))
        elif file_name.startswith('stan'):
            stan_data.append(extracted_rows.reshape(-1, extracted_rows.shape[1] * extracted_rows.shape[2]))
        
        # Display or use the loaded data and metadata as needed
        print(f'Loaded data from file: {file_name}')
        print(f'Loaded timestamps from file: {file_name}')
        print(f'Loaded metadata from file: {file_name}')
    
    except Exception as e:
        print(f'Error processing file {file_name}: {str(e)}')
        continue  # Skip to the next file

# Concatenate the data arrays
happy_data = np.concatenate(happy_data, axis=0) if happy_data else np.array([])
stan_data = np.concatenate(stan_data, axis=0) if stan_data else np.array([])

# Display sizes of concatenated data
print(f'Concatenated Happy Data Size: {happy_data.shape}')
print(f'Concatenated Stan Data Size: {stan_data.shape}')
