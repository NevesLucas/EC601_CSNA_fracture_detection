from PIL import Image
import pandas as pd
import pydicom
import numpy as np
import cv2
import os

RSNA_2022_PATH = 'datasets/input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'

def load_metadata():
    # Load metadata
    train_df = pd.read_csv(RSNA_2022_PATH + "/train.csv")
    train_bbox = pd.read_csv(RSNA_2022_PATH + "/train_bounding_boxes.csv")
    test_df = pd.read_csv(RSNA_2022_PATH + "/test.csv")
    ss = pd.read_csv(RSNA_2022_PATH + "/sample_submission.csv")

    # https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344862
    bad_scans = ['1.2.826.0.1.3680043.20574','1.2.826.0.1.3680043.29952']

    for uid in bad_scans:
        train_df.drop(train_df[train_df['StudyInstanceUID']==uid].index, axis=0, inplace=True)
    return [train_df, train_bbox, test_df, ss]

"""
image loading function for rsna competition images, stored as 'dicom' images
"""
def load_dicom(path):
    img = pydicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

