import sys, os, csv, cv2
import pandas as pd
from sklearn.model_selection import train_test_split

video_folder = os.getcwd() + '/data/videos/'
image_folder = os.getcwd() + '/data/images/'
split_folder = os.getcwd() + '/data/splits/'

def getFrame(vid, sec, path, filename, count):
  vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
  hasFrames, image = vid.read()
  if hasFrames:
    cv2.imwrite(image_folder + filename + "_frame"+str(count)+".jpg", image)
    return hasFrames

def createCSVFile(row):
    video = row[0]
    frameRate = 0.5 
    count = 1
    sec = 0
    if os.path.exists(image_folder + video + '_frame1.jpg'):
        print("Image frames already exists for video {}".format(video))
    else:
        print("Separating video {} into frames.".format(video))
        vidcap = cv2.VideoCapture(video_folder + video)
        success = getFrame(vidcap, sec, video_folder, video, count)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(vidcap, sec, video_folder, video, count)

# Splitting the data into train, validation, and test subsets with a ratio of 8:1:1
# If video hasn't been separated into frames, this is done in the process

df = pd.read_csv(video_folder + 'videos.csv')
train, remain = train_test_split(df, test_size=0.2, random_state=1)
test, val = train_test_split(remain, test_size=0.5, random_state=1)

for arg in sys.argv:
    if not arg == os.path.basename(__file__):
        file_name = arg.rsplit('=', 1)[1]
        with open(split_folder + file_name, 'w') as csv_file:
            filewriter = csv.writer(csv_file, delimiter=',')
            filewriter.writerow(['image_id','deepfake','original_video','dataset'])
            if file_name.startswith('train'):
                for index, row in train.iterrows():
                    createCSVFile(row)
                    for img in os.listdir(image_folder):
                        if img.startswith(row[0]):
                            filewriter.writerow([img,row[1],row[0],row[2]])
            elif file_name.startswith('val'):
                for index, row in val.iterrows():
                    createCSVFile(row)
                    for img in os.listdir(image_folder):
                        if img.startswith(row[0]):
                            filewriter.writerow([img,row[1],row[0],row[2]])
            elif file_name.startswith('test'):
                for index, row in test.iterrows():
                    createCSVFile(row)
                    for img in os.listdir(image_folder):
                        if img.startswith(row[0]):
                            filewriter.writerow([img,row[1],row[0],row[2]])

print('Train, Validation, and Test splits have been created.')