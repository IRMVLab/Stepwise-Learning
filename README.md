# Step-wise-Learning


Train.py is used to train the simple feedforward network using step-wise learning approach.
- train.py might need several attempts to converge. 
- this might takes hours to converge on initial pair of data. 

angle_measurement.py is used to evaluate the performance of a network on various angles of an object.


Dataset Preparation:
----------------------
Each of 'training_set' and 'validation_set' directories should contain sub-directories. Where each sub-directory should have multiple images of same scene or class. The images belonging to same class or same scene should be having modified environmental conditions like some images taken from front view and some from different viewing angles or in different illumination coditions. The images may be scaled, rotated or partially occluded.


To Train Embeddings Generator:
---------------------------
Test Environment:
- Machine: Acer Predator Helios 300 (PH315-53-72XD)
- Ubuntu 20.04
- GPU: RTX 2060 - 6 GB
- 16 GB RAM
- Tensorflow/ Keras 2.1.0
- Cuda 10.2
- Cudnn 8.0.1



To run the Code:
1. Change 'work_dir' and 'dataset_path' to your checkpoint location and dataset location.
2. Run 'train.py'


For Binary classification on smaller Datasets:
----------------------------------------------
The core training approach proposed in the paper "Viewpoint-Invariant Loop Closure Detection using step-wise Learning with Controlling Embeddings of Landmarks" is used for binary classification purpose in 'binary_classifier.py'. For Core paper topic, use 'train.py'.

Install pynput (optional). If you do not want to install, just comment out the lines needing this library.

Test Environment:
- Machine: Acer Predator Helios 300 (PH315-53-72XD)
- Ubuntu 20.04
- GPU: RTX 2060 - 6 GB
- 16 GB RAM
- Tensorflow 2.5.0
- Keras 2.6.0
- Numpy 1.19.5
- Cuda 11.2
- Cudnn 8
- GCC 9.3.0

To run the code:

1. Change 'work_dir' and 'dataset_path' in binary_classification.py
2. Change 'task' to number of samples, you want to train. Currently set to 10, makes 5 image pairs for training purpose.
3. 'work_dir' = path to save checkpoints as well as log information
4. 'dataset_path' = path to main directory, where there should be 'training_set' and 'validation_set' folders
5. Each of 'training_set' or 'validation_set' folders should have separate directories for each class.
6. run 'binary_classifier.py' and wait for training to converge.
7. It will train multiple times, may be the features used first time, does not fit to other image pairs, so this data will be discarded and another features will be considered from first pair and the network will try to optimize it until the required 'loss_th', 'vloss_th', 'acc_th', 'vacc_th' are obtained.
