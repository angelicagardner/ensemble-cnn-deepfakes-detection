Pre-trained models provided by the original authors of each single model.
Download them and place them in this folder (models/pre_trained/) as follows:

1. Capsule -> models/pre_trained/capsule_21.pt

https://github.com/nii-yamagishilab/Capsule-Forensics-v2/blob/master/checkpoints/binary_faceforensicspp/capsule_21.pt

2. DSP-FWA -> models/pre_trained/SPP-res50.pth

https://drive.google.com/file/d/13wbA5kHRGODBDdiJ2gPeB1XK4KiCh-Im/view
(if this link no longer works, see https://github.com/danmohaha/DSP-FWA for the current link)

3. Ictu Oculi (CNN-VGG16, TensorFlow checkpoint) -> models/pre_trained/ictu_oculi/
   (the checkpoint files from the authors' ckpt_CNN folder: checkpoint, *.index, *.data-*)

CNN-VGG16:
https://drive.google.com/file/d/1NJ160vkLq8JCVTZC8ocsDD8VfFskDSRM/view
(if this link no longer works, see https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi for the current link)

The TensorFlow checkpoint is converted to the PyTorch version of the network (models/Ictu_Oculi.py)
when it is loaded. The LRCN-VGG16 model is not used in this experiment.

4. XceptionNet -> models/pre_trained/all_c23.p

http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip
(the face-based XceptionNet model; the unpacked zip can also be placed in this folder,
 the file faceforensics++_models_subset/face_detection/xception/all_c23.p is then found automatically)

In addition, the Capsule model downloads the ImageNet weights for its VGG19 feature extractor
through torchvision the first time it is used.
