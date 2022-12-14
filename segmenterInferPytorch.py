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
from monai.transforms import AsDiscrete

from torchvision.ops import masks_to_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch.cuda.amp as amp
import torchio as tio

with open('config.json', 'r') as f:
    paths = json.load(f)

cachedir = paths["CACHE_DIR"]
modelWeights = paths["seg_weights"]

segModel = torch.load(modelWeights, map_location="cpu") # need 2 gpus for this workflow
segModel.eval()
segResize = tio.Resize((128, 128, 200)) #resize for segmentation
classResize = tio.Resize((256,256,256))

def boundingVolume(pred,original_dims):
    #acquires the 3d bounding rectangular prism of the segmentation mask
    indices = torch.nonzero(pred)
    min_indices, min_val = indices.min(dim=0)
    max_indices, max_val = indices.max(dim=0)
    print(min_indices)
    print(max_indices)
    return (min_indices[1].item(), original_dims[0]-max_indices[1].item(),
            min_indices[2].item(), original_dims[1]-max_indices[2].item(),
            min_indices[3].item(), original_dims[2]-max_indices[3].item())

def cropData(dataElement):
    downsampled = segResize(dataElement)
    originalSize = dataElement[0].size()
    rescale = tio.Resize(originalSize)
    mask = segModel(downsampled.unsqueeze(0))
    mask = torch.argmax(mask, dim=1)
    mask = rescale(mask)
    bounding_prism = boundingVolume(mask,originalSize)
    crop = tio.Crop(bounding_prism)
    cropped = crop(dataElement)
    return classResize(cropped)

smartCrop = tio.Lambda(cropData)

root_dir="./"

if torch.cuda.is_available():
     print("GPU enabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = kaggleDataLoader.KaggleDataLoader()

train, val = dataset.loadDatasetAsClassifier(trainPercentage = 1.0, train_aug=smartCrop)

basic_sample = train[10]
# get original dims first

fig, ax = plt.subplots()
ims = []
for sagittal_slice_tensor in basic_sample.ct.data[0]:

    im = ax.imshow(sagittal_slice_tensor.detach().numpy(), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()