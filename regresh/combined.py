import csv

spreadsheet_filenames = [
'stacked_total_image_data_one.csv',
'stacked_total_image_data_two.csv',
'stacked_total_image_data_three.csv',
'stacked_total_image_data_four.csv',
'stacked_total_image_data_five.csv',
'stacked_total_image_data_six.csv',
]

for filename in spreadsheet_filenames:
    with open(filename, 'r') as csvfile:
        sales = csv.reader(csvfile)
        for row in sales:
            print (row)