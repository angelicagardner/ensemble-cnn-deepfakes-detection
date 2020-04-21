import sys, os, csv
import pandas as pd
from sklearn.model_selection import train_test_split

data_folder = os.getcwd() + '/data/videos/'
split_folder = os.getcwd() + '/data/splits/'

# Splitting the data into train, validation, and test subsets with a ratio of 8:1:1
df = pd.read_csv(data_folder + 'videos.csv')
train, remain = train_test_split(df, test_size=0.2, random_state=1)
test, val = train_test_split(remain, test_size=0.5, random_state=1)

for arg in sys.argv:
    if not arg == os.path.basename(__file__):
        file_name = arg.rsplit('=', 1)[1]
        with open(split_folder + file_name, 'w') as csv_file:
            filewriter = csv.writer(csv_file, delimiter=',')
            filewriter.writerow(['video_id','fake','original_dataset'])
            if file_name.startswith('train'):
                for index, row in train.iterrows():
                    filewriter.writerow(row)
            elif file_name.startswith('val'):
                for index, row in val.iterrows():
                    filewriter.writerow(row)
            elif file_name.startswith('test'):
                for index, row in test.iterrows():
                    filewriter.writerow(row)

print('Train, Validation, and Test splits have been created.')