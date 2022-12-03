import kaggleDataLoader
import json

from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from monai.data import decollate_batch, DataLoader,Dataset,ImageDataset
from monai.metrics import DiceMetric
from monai.losses.dice import DiceLoss
from monai.networks.nets import BasicUNet
from monai.visualize import plot_2d_or_3d_image

from torchvision.ops import masks_to_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch.cuda.amp as amp
import torchio as tio

with open('config.json', 'r') as f:
    paths = json.load(f)

def boundingVolume(pred):
    #acquires the 3d bounding rectangular prism of the segmentation mask
    pred = (pred > 0.2).float()
    indices = torch.nonzero(pred)
    min_indices = indices.min(dim=0)[0]
    max_indices = indices.max(dim=0)[0]
    print(min_indices)
    print(max_indices)


cachedir = paths["CACHE_DIR"]
modelWeights = paths["seg_weights"]

root_dir="./"

if torch.cuda.is_available():
     print("GPU enabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = kaggleDataLoader.KaggleDataLoader()
train, val = dataset.loadDatasetAsClassifier(trainPercentage = 1.0)

model = torch.load(modelWeights, map_location=torch.device('cpu'))
model.eval()

downsample = tio.Resample(1)
cropOrPad = tio.CropOrPad((128, 128, 200))

spatial_process = tio.Compose([downsample,cropOrPad])
basic_sample = train[10]

downsampled = spatial_process(basic_sample)

reverseTransform = downsampled.get_inverse_transform(image_interpolation='linear')

prediction = model(downsampled.ct.data.unsqueeze(0) )  # just test on first image
prediction = torch.argmax(prediction, dim=0)
# native_prediction = reverseTransform(prediction[0])
fig, ax = plt.subplots()
ims = []
for sagittal_slice_tensor in prediction[0][0]:
    im = ax.imshow(sagittal_slice_tensor.detach().numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

boundingVolume(prediction[0][0])
