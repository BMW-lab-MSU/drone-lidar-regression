
import pandas as pd
import os

# Specify the path to the folder
folder_path = 'training_hdf'

# List all files in the folder
files = os.listdir(folder_path)

# Count the number of files
num_files = len(files)

print(f"There are {num_files} files in the folder.")

# Load the CSV file
df = pd.read_csv('rpm1_values.csv')

# Get the dimensions of the DataFrame
rows, columns = df.shape

print(f"Rows: {rows}, Columns: {columns}")