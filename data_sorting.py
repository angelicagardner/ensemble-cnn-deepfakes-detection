import cv2
import os, csv
import shutil
import numpy as np

##########
# Put all videos in the same folder and create CSV-file
##########

path = os.getcwd()

# Replace these folder locations with where you're storing your datasets
dataset_celebdf = path + '/celeb-df'
dataset_deepfakedetection = path + '/deepfakedetection'
dataset_deepfakedetectionchallenge = path + '/deepfake-detection-challenge'

datasets = [dataset_celebdf, dataset_deepfakedetection, dataset_deepfakedetectionchallenge]

output_folder = path + '/data/videos/'

with open(output_folder + 'videos.csv', 'a') as csv_file:
  filewriter = csv.writer(csv_file, delimiter=',')
  filewriter.writerow(['video_id','fake','original_dataset'])

for dataset in datasets:
  for real_video in os.listdir(dataset + '/real'):
    if real_video.endswith('.mp4'):
      shutil.move(dataset + '/real/' + real_video, output_folder + real_video)
      with open(output_folder + 'videos.csv', 'a') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=',')
        filewriter.writerow([real_video, 0, dataset.rsplit('/', 1)[1]])
    else:
      continue
  for fake_video in os.listdir(dataset + '/fake'):
    if fake_video.endswith('.mp4'):
      shutil.move(dataset + '/fake/' + fake_video, output_folder + fake_video)
      with open(output_folder + 'videos.csv', 'a') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=',')
        filewriter.writerow([fake_video, 1, dataset.rsplit('/', 1)[1]])
    else:
      continue

print("Files have been successfully moved into the output folder and all information put into a CSV file.")