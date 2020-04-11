# 2dv50e
This project contains the source code of all experiments described in 'Degree Project at Bachelor Level - VT2020 - Linnaeus University.'
## Abstract
> ...
## 1. Project folder structure
- Datasets folder, where you can place your training, evaluation, and test sets:
```./data ```
- Pre-trained models folder, where the fitted models trained with its default settings described in the associated paper are provided here:
```
      ./models
```
## 2. Datasets
Each dataset has two parts:
- Real videos: ./data/<dataset_name>/<train;validation;test>/0_real_videos
- Real images: ./data/<dataset_name>/<train;test;validation>/0_real_images
- Fake videos: ./data/<dataset_name>/<train;test;validation>/1_fake_videos
- Fake images: ./data/<dataset_name>/<train;test;validation>/1_fake_images

**Note**: When videos need to be separated into frames (images) the script *preprocessing.py* can be used.
## 3. Training
...
## 4. Evaluating
...
### Requirements
See file *requirements.txt*
## Cite
This research was carried out while the author studied at Linneaus University, Sweden.
If you use this code or research as a reference, please cite:
```
...
```