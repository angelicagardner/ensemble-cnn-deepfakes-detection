import cv2
import os
import shutil
import numpy as np

##########
# Split a dataset into train, validation, and test
##########

dataset = os.getcwd() + '/data' + '/celeb-df'
# dataset = os.getcwd() + '/data' + '/deepfakedetection'

train_set_real = dataset + '/train/0_real'
train_set_fake = dataset + '/train/1_fake'
val_set_real = dataset + '/validation/0_real'
val_set_fake = dataset + '/validation/1_fake'
test_set_real = dataset + '/test/0_real'
test_set_fake = dataset + '/test/1_fake'

original_folder_real = dataset + '/videos/real' # Replace this folder location with where you've stored the full video dataset of real videos
original_folder_fake = dataset + '/videos/fake' # Replace this folder location with where you've stored the full video dataset of fake videos

x_real = np.arange(len(os.listdir(original_folder_real)))
x_fake = np.arange(len(os.listdir(original_folder_fake)))
np.random.shuffle(x_real)
np.random.shuffle(x_fake)
train_real, validation_real, test_real = np.split(x_real, [int(.8 * len(x_real)), int(.9 * len(x_real))])
train_fake, validation_fake, test_fake = np.split(x_fake, [int(.8 * len(x_fake)), int(.9 * len(x_fake))])

for v in train_real:
  video = os.listdir(original_folder_real)[v]
  shutil.copyfile(original_folder_real + '/' + video, train_set_real + '/' + video)
for v in train_fake:
  video = os.listdir(original_folder_fake)[v]
  shutil.copyfile(original_folder_fake + '/' + video, train_set_fake + '/' + video)

for v in validation_real:
  video = os.listdir(original_folder_real)[v]
  shutil.copyfile(original_folder_real + '/' + video, val_set_real + '/' + video)
for v in validation_fake:
  video = os.listdir(original_folder_fake)[v]
  shutil.copyfile(original_folder_fake + '/' + video, val_set_fake + '/' + video)

for v in test_real:
  video = os.listdir(original_folder_real)[v]
  shutil.copyfile(original_folder_real + '/' + video, test_set_real + '/' + video)
for v in test_fake:
  video = os.listdir(original_folder_fake)[v]
  shutil.copyfile(original_folder_fake + '/' + video, test_set_fake + '/' + video)

print("Files have been successfully separated into the three datasets and moved into their respective folders.")

##########
# Separate videos into image frames
##########

frameRate = 0.1 

def getFrame(vid, sec, path, filename):
  vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
  hasFrames, image = vid.read()
  if hasFrames:
    cv2.imwrite(path + '/frames/' + filename + "_frame"+str(count)+".jpg", image)
    return hasFrames

for subdir_level_one in os.listdir(dataset):
  subdir_path_level_one = dataset + '/' + subdir_level_one
  if os.path.isdir(subdir_path_level_one):
    for subdir_level_two in os.listdir(subdir_path_level_one):
      subdir_path_level_two = subdir_path_level_one + '/' + subdir_level_two
      if os.path.isdir(subdir_path_level_two):
        for video in os.listdir(subdir_path_level_two):
          if video.endswith('.mp4'):
            count = 1
            sec = 0
            vidcap = cv2.VideoCapture(subdir_path_level_two + '/' + video)
            success = getFrame(vidcap, sec, subdir_path_level_two, video)
            while success:
              count = count + 1
              sec = sec + frameRate
              sec = round(sec, 2)
              success = getFrame(vidcap, sec, subdir_path_level_two, video)

print("Videos have been successfully separated into image frames and all the image frames have been saved in their respective folder locations.")