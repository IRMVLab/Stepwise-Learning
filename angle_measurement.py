
import torchvision.models as models
import torch
from torch import nn
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from torch.autograd import Variable
from PIL import Image
from torchsummary import summary
from skimage.transform import resize
import numpy as np
import os, sys, math
import scipy.io as sio
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import distance
import cv2







# alexnet_model = models.alexnet(pretrained=True)
# #print(alexnet_model)
# summary(alexnet_model, (3, 224, 224))
# alexnet_model2 = nn.Sequential(alexnet_model.features[0:7])

alexnet_model = models.resnext101_32x8d(pretrained=True)
alexnet_model2 = nn.Sequential(*list(alexnet_model.children())[:8])[:-1]
# Feature List
feature_list = []


dataset_path = "F:/Research_heavyData/Datasets/angle/obj10/"       # Path to directory containing single object with different angles

file_names = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".ppm"))]
file_names.sort()


# Confusion Matrix
#loop_closure_filenames = np.zeros((len(file_names), len(file_names)), dtype=float)
loop_closure_filenames = np.zeros((1, len(file_names)), dtype=float)

for f in file_names:
    print("Reading file {}".format(f))
    images = torch.Tensor(1, 3, 224, 224)
    #img = Image.open(f)
    img = cv2.imread(f)
    img = img_to_array(img)
    img = resize(img, (224, 224, 3), anti_aliasing=True)
    #img = img.convert('1')
    data = np.array( img, dtype='uint8' )
    data=np.rollaxis(data, 2, 0)
    # print(data)
    images[0] = torch.from_numpy(data)
    tensor = Variable(images)
    # print(tensor)
    
    #outputs = alexnet_model2.forward(tensor)
    outputs = alexnet_model2(tensor)
    #print("output: {}".format(outputs.data.shape))
    _, predicted = torch.max(outputs.data, 1)
    #print(predicted)
    feature_list.append(np.array(predicted).flatten())
    
print("Fetaures collected")


#for i in range(len(feature_list)):
for j in range(len(feature_list)):
    #score=cosine_similarity(feature_list[i], feature_list[j])
    score = distance.euclidean(feature_list[0], feature_list[j])
    print("Image {} and image {}: {}".format(0,j, score))
    loop_closure_filenames[0, j] = score
        
sio.savemat(os.path.splitext(sys.argv[0])[0] + '_obj10_L2_distance_conf_matrix_conv3.mat', {'truth4':loop_closure_filenames})
plt.imshow(loop_closure_filenames, extent=[0,len(file_names)-1,len(file_names)-1,0], cmap='gray')
plt.colorbar()
plt.show(block=True)
print("Done")
