import csv, os, shutil

##########
# Put all videos in the same folder and create CSV-file
##########

# The datasets are placed in the data folder, each with one sub-folder for real videos
# and one for deepfake videos: data/<dataset>/real/ and data/<dataset>/fake/
data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Replace these folder locations with where you're storing your datasets
dataset_celebdf = os.path.join(data_folder, "celeb-df")
dataset_deepfakedetection = os.path.join(data_folder, "deepfakedetection")
dataset_deepfakedetectionchallenge = os.path.join(
    data_folder, "deepfake-detection-challenge"
)

datasets = [
    dataset_celebdf,
    dataset_deepfakedetection,
    dataset_deepfakedetectionchallenge,
]

output_folder = os.path.join(data_folder, "videos") + "/"

with open(output_folder + "videos.csv", "w", newline="") as csv_file:
    filewriter = csv.writer(csv_file, delimiter=",")
    filewriter.writerow(["video_id", "fake", "original_dataset"])

for dataset in datasets:
    for label, subfolder in ((0, "real"), (1, "fake")):
        for video in sorted(os.listdir(os.path.join(dataset, subfolder))):
            if video.endswith(".mp4"):
                shutil.move(
                    os.path.join(dataset, subfolder, video), output_folder + video
                )
                with open(output_folder + "videos.csv", "a", newline="") as csv_file:
                    filewriter = csv.writer(csv_file, delimiter=",")
                    filewriter.writerow([video, label, os.path.basename(dataset)])

print(
    "Files have been successfully moved into the output folder and all information put into a CSV file."
)
