import cv2
import os
from shutil import copyfile
import numpy as np

# Shuffle the videos for a dataset and split it into train, validation, and test
dataset = os.getcwd() + '/data' + '/celeb-df'
train_set = dataset + '/train/1_fake'
val_set = dataset + '/validation/1_fake'
test_set = dataset + '/test/1_fake'

x = np.arange(len(os.listdir(dataset + '/videos/fake')))
np.random.shuffle(x)
train, validation, test = np.split(x, [int(.8 * len(x)), int(.9 * len(x))])

for v in train:
  video = os.listdir(dataset + '/videos/fake')[v]
  copyfile(dataset + '/videos/fake/' + video, train_set + '/' + video)

for v in validation:
  video = os.listdir(dataset + '/videos/fake')[v]
  copyfile(dataset + '/videos/fake/' + video, val_set + '/' + video)

for v in test:
  video = os.listdir(dataset + '/videos/fake')[v]
  copyfile(dataset + '/videos/fake/' + video, test_set + '/' + video)

# 2. Separate each video into frames

#vidcap = cv2.VideoCapture('id0_id4_0005.mp4')

#def getFrame(sec):
  #vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
  #hasFrames, image = vidcap.read()
  #if hasFrames:
    #cv2.imwrite("image"+str(count)+".jpg", image) # save frame as JPG fle
    #return hasFrames

sec = 0
frameRate = 0.1 
count = 1
#success = getFrame(sec)
#while success:
  #count = count + 1
  #sec = sec + frameRate
  #sec = round(sec, 2)
  #success = getFrame(sec)