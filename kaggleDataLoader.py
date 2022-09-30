
from PIL import Image

import pandas as pd
import pydicom
import nibabel as nib
import numpy as np
import cv2
import os

RSNA_2022_PATH    = 'datasets/input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH  = f'{RSNA_2022_PATH}/test_images'

class KaggleDataLoader:

    def __init__(self):
        # Load metadata

        self.trainImagePath = TRAIN_IMAGES_PATH
        self.segPath        = os.path.join(TRAIN_IMAGES_PATH,          "segmentations")
        self.trainDf        = pd.read_csv(os.path.join(RSNA_2022_PATH, "train.csv"))
        self.trainBbox      = pd.read_csv(os.path.join(RSNA_2022_PATH, "train_bounding_boxes.csv"))
        self.testDf         = pd.read_csv(os.path.join(RSNA_2022_PATH, "test.csv"))
        self.ss             = pd.read_csv(os.path.join(RSNA_2022_PATH, "sample_submission.csv"))

        # https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344862
        bad_scans           = ['1.2.826.0.1.3680043.20574', '1.2.826.0.1.3680043.29952']
        for uid in bad_scans:
            self.trainDf.drop(self.trainDf[self.trainDf['StudyInstanceUID'] == uid].index, axis = 0, inplace = True)

    def bboxFromIndex(self, id, sliceNum):
        box = self.trainBbox.loc[(self.trainBbox.StudyInstanceUID == id & self.trainBbox.slice_number == sliceNum), :]
        return list(box.values[0])

    def fracturedBones(self, patientID):
        fractured_bones = []
        temp = self.trainDf.loc[self.trainDf.StudyInstanceUID == patientID, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']]
        temp = list(temp.values[0])  # there is one row per id
        for i in range(len(temp)):
            if temp[i] == 1:
                fractured_bones.append('C' + str(i + 1))
        return fractured_bones

    """
    image loading function for rsna competition images, stored as 'dicom' images
    """

    def loadDicom(self, path):
        img = pydicom.dcmread(path)
        img.PhotometricInterpretation = 'YBR_FULL'
        data = img.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

    def loadSliceFromId(self, patientID, sliceIndex):
        targetPath = os.path.join(self.trainImagePath, patientID, str(sliceIndex), ".dcm")
        return self.loadDicom(targetPath)

    def loadSegmentationsForPatient(self, patientID):
        segmentations = nib.load(os.path.join(self.segPath, patientID, '.nii')).get_fdata()
        segmentations = segmentations[:, ::-1, ::-1]
        segmentations = segmentations.transpose(2, 1, 0)
        return segmentations
    
    def loadDataset(self):
        """
        prepare full dataset for training
        """
