import h5py
import numpy as np
import os
import psutil

# Define memory logging function
def log_memory_usage():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB
    print(f"Memory usage: {memory_usage:.2f} GB")

# Call the function at the start
log_memory_usage()

data_folder = 'training_hdf'
batch_size = 1000  # Adjust this based on your memory limits
output_csv_path = 'stacked_total_image_data.csv'

# Initialize variables
temp_array = []
first_image = True

# Process files and images
for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            print(f'Processing file: {filename}')  
            with h5py.File(filepath, 'r') as file:
                images = file['data']['data'][:]
                log_memory_usage()  # Log memory usage after loading images

                # Process images in batches
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    
                    # Stack images to temporary array
                    if first_image:
                        temp_array = batch_images
                        first_image = False
                    else:
                        temp_array = np.vstack((temp_array, batch_images))
                    
                    log_memory_usage()  # Log memory usage after processing a batch
                
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

if temp_array:
    # Save the final stacked array to CSV
    np.savetxt(output_csv_path, temp_array, delimiter=',')
    print(f'Final array shape: {temp_array.shape}')
else:
    print('No valid arrays found in temp_array.')

print('Processing complete.')
