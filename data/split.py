import sys, os, csv, cv2, dlib
import pandas as pd
from sklearn.model_selection import train_test_split

video_folder = os.getcwd() + "/data/videos/"
image_folder = os.getcwd() + "/data/images/"
split_folder = os.getcwd() + "/data/splits/"

face_detector = dlib.get_frontal_face_detector()


# Version of getFrame() function that crops facial area
def getFrame(vid, sec, filename, count):
    vid.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vid.read()
    if hasFrames:
        faces = face_detector(image, 1)
        print("Number of faces detected: {}".format(len(faces)))
        for i, d in enumerate(faces):
            # Keep the face area inside the image
            top, bottom = max(d.top(), 0), min(d.bottom(), image.shape[0])
            left, right = max(d.left(), 0), min(d.right(), image.shape[1])
            cropped_face = image[top:bottom, left:right]
            if cropped_face.size == 0:
                continue
            # In case more than one face is detected, all faces are saved as video frames
            face_suffix = "" if i == 0 else "_face" + str(i + 1)
            cv2.imwrite(
                image_folder + filename + "_frame" + str(count) + face_suffix + ".jpg",
                cropped_face,
            )
    return hasFrames


# Version of getFrame() function that keeps full video frame
"""
def getFrame(vid, sec, filename, count):
  vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
  hasFrames, image = vid.read()
  if hasFrames:
    cv2.imwrite(image_folder + filename + "_frame"+str(count)+".jpg", image)
  return hasFrames
"""


# Separates a video into frames: one frame from every 0.5 seconds of the video
def separateIntoFrames(video, existing_images):
    frameRate = 0.5
    count = 1
    sec = 0
    if video in existing_images:
        print("Image frames already exists for video {}".format(video))
    else:
        print("Separating video {} into frames.".format(video))
        vidcap = cv2.VideoCapture(video_folder + video)
        success = getFrame(vidcap, sec, video, count)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(vidcap, sec, video, count)


# Groups the images in the image folder by the video they come from
def imagesPerVideo():
    images = {}
    for img in sorted(os.listdir(image_folder)):
        if "_frame" in img:
            images.setdefault(img.rsplit("_frame", 1)[0], []).append(img)
    return images


if not os.path.exists(split_folder):
    os.makedirs(split_folder)

# Splitting the data into train, validation, and test subsets with a ratio of 8:1:1
# (the videos are split before they are separated into frames so frames from one video are only in one subset)
df = pd.read_csv(video_folder + "videos.csv")
train, remain = train_test_split(df, test_size=0.2, random_state=1)
test, val = train_test_split(remain, test_size=0.5, random_state=1)
subsets = {"train": train, "val": val, "test": test}

# Arguments: train_csv=<file name> val_csv=<file name> test_csv=<file name>
for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    subset_name = arg.split("=", 1)[0].replace("_csv", "")
    file_name = arg.rsplit("=", 1)[1]
    if subset_name not in subsets:
        continue
    subset = subsets[subset_name]

    # If a video hasn't been separated into frames, this is done in the process
    existing_images = imagesPerVideo()
    for index, row in subset.iterrows():
        separateIntoFrames(row["video_id"], existing_images)
    images = imagesPerVideo()

    with open(split_folder + file_name, "w", newline="") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=",")
        filewriter.writerow(["frame_id", "deepfake", "original_video", "dataset"])
        for index, row in subset.iterrows():
            for img in images.get(row["video_id"], []):
                filewriter.writerow(
                    [img, row["fake"], row["video_id"], row["original_dataset"]]
                )

print("Train, Validation, and Test splits have been created.")
