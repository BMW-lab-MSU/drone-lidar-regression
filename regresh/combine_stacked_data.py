# import pandas as pd

f1 = pd.read_csv('stacked_total_image_data_one.csv', header=None)
f2 = pd.read_csv('stacked_total_image_data_two.csv', header=None)
f3 = pd.read_csv('stacked_total_image_data_three.csv', header=None)
f4 = pd.read_csv('stacked_total_image_data_four.csv', header=None)
f5 = pd.read_csv('stacked_total_image_data_five.csv', header=None)
f6 = pd.read_csv('stacked_total_image_data_six.csv', header=None)

merged = pd.concat(f1, f2, f3, f4, f5, f6)
 

merged.to_csv('merged.csv', index=None, header=None)





