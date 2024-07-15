import pandas as pd

# Load the CSV file
df = pd.read_csv('stacked_total_image_data.csv')

# Check the dimensions of the DataFrame
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Print the first few rows to check the data
print(df.head())