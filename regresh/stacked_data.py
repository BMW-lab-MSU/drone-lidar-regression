import h5py
import numpy as np
import os
import psutil

def log_memory_usage():
    process = psutil.Process()
    memory_usage_gb = process.memory_info().rss / (1024 ** 3)
    print(f"Memory usage: {memory_usage_gb:.2f} GB")
    return memory_usage_gb

data_folder = 'training_hdf'
output_array = []
memory_limit_gb = 29.5  # Memory limit in GB

for filename in os.listdir(data_folder):
    if filename.endswith('.hdf5'):
        filepath = os.path.join(data_folder, filename)
        
        try:
            print(f'Processing file: {filename}')
            with h5py.File(filepath, 'r') as file:
                
                images = file['data']['data'][:]

                for image in images:
                    output_array.append(image)
        
        except MemoryError as mem_err:
            print(str(mem_err))
            break
        except Exception as e:
            print(f'Failed to process file {filename}: {str(e)}')

if output_array:
    final_array = np.vstack(output_array)
    print(f'Final array shape: {final_array.shape}')
    
    np.savetxt('stacked_total_image_data.csv', final_array, delimiter=',')
else:
    print('No valid arrays found in output_array.')

print('Processing complete.')
