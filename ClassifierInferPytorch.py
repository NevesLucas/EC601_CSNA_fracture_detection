import json
import torch
import torch.nn as nn
import torchio as tio
import pandas as pd
from rsna_cropped import RSNACervicalSpineFracture

from sklearn.metrics import classification_report
with open('config.json', 'r') as f:
    paths = json.load(f)


if torch.cuda.is_available():
     print("GPU enabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RSNA_2022_PATH = paths["RSNA_2022_PATH"]
cachedir = paths["CACHE_DIR"]
segWeights = paths["seg_weights"]
classWeights = paths["classifier_weights"]
segModel = torch.load(segWeights, map_location="cpu") # need 2 gpus for this workflow
segModel.eval()
classModel = torch.load(classWeights, map_location=device) # need 2 gpus for this workflow
classModel.eval()

segResize = tio.Resize((128, 128, 200)) #resize for segmentation
classResize = tio.Resize((256,256,256))

pred_cols = [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "patient_overall"
]

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

HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900
clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
rescale = tio.RescaleIntensity(percentiles=(0.5, 99.5))
preprocess_intensity = tio.Compose([
    clamp,
    rescale,
])
normalize_orientation = tio.ToCanonical()

preprocess_spatial = tio.Compose([
    normalize_orientation])

preprocess = tio.Compose([
    preprocess_spatial,
    preprocess_intensity,
    smartCrop
])

root_dir="./"

#trainSet = tio.datasets.RSNACervicalSpineFracture(RSNA_2022_PATH, add_segmentations=False)
trainSet = RSNACervicalSpineFracture(RSNA_2022_PATH, add_segmentations=False)
with torch.no_grad():
    predicted = []
    actual = []

    for classifier_input in trainSet:
        # get original dims first
        #classifier_input = preprocess(samples)
        logits = classModel(classifier_input.ct.data.unsqueeze(0))[0]
        gt = [classifier_input[target_col] for target_col in pred_cols]
        sig = nn.Sigmoid()
        preds = sig(logits)
        overall = preds.numpy().squeeze()
        multiclass_pred = (overall > 0.5)*1
        predicted.append(multiclass_pred)
        actual.append(gt)
        print(multiclass_pred)
    report = classification_report(predicted, actual, output_dict=True,
    target_names=pred_cols)
    print(report)
    df = pd.DataFrame(report).transpose()
    df.to_csv("modelReport.csv")
    # get accuracy metrics

