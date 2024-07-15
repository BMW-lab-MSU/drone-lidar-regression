import h5py
import numpy as np
import os
import shutil
import random

splitting_data = 'recopied_h5s'

training_hdf = '/home/j19n522/regresh/training_hdf'
testing_hdf = '/home/j19n522/regresh/testing_hdf'

os.makedirs(training_hdf, exist_ok = True)
os.makedirs(testing_hdf, exist_ok = True)

files = [f for f in os.listdir(splitting_data) if f.endswith('.hdf5')]

random.shuffle(files)

split_point = int(len(files) * 0.8)

train_files = files[:split_point]
test_files = files[split_point:]

for file in train_files:
    shutil.move(os.path.join(splitting_data, file), os.path.join(training_hdf, file))

for file in test_files:
    shutil.move(os.path.join(splitting_data, file), os.path.join(testing_hdf, file))

print(f'Moved {len(train_files)} files to {training_hdf}')
print(f'Moved {len(test_files)} files to {testing_hdf}')
print('Processing complete.')
