import os
import h5py
import pandas as pd

def hdf5_to_dict(file_name, verbose=True):
    data = {}

    def load_content(group, path='/'):
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                var_name = key.replace(' ', '_')
                if item.shape == ():
                    data[path + var_name] = item[()]
                else:
                    data[path + var_name] = item[:]
            elif isinstance(item, h5py.Group):
                load_content(item, path + key + '/')

        for key, attr in group.attrs.items():
            attr_name = key.replace(' ', '_')
            data[path + attr_name] = attr

    try:
        with h5py.File(file_name, 'r') as f:
            load_content(f)
    except Exception as e:
        if verbose:
            print(f"Error reading {file_name}: {e}")

    return data

def process_all_hdf5(data_dir):
    all_data = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.hdf5'):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")
            try:
                data = hdf5_to_dict(file_path)
                all_data.append(data)
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")

    if all_data:
        combined_data = pd.DataFrame(all_data)
        return combined_data
    else:
        print("No data was read from the HDF5 files.")
        return pd.DataFrame()

if __name__ == "__main__":
    data_dir = 'training_data/'
    
    if os.path.exists(data_dir):
        combined_data = process_all_hdf5(data_dir)

        if not combined_data.empty:
            print(combined_data)
            combined_data.to_csv(os.path.join(data_dir, 'combined_data.csv'), index=False)
        else:
            print("No data to save.")
    else:
        print(f"Directory {data_dir} not found.")