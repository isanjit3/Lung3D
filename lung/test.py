
import os
import config
import numpy as np
from unet3d import dicom_data_util
from unet3d.utils import dicom_util

def display_img_from_hdf5():
    hdf5 = dicom_data_util.open_data_file(config.config["data_file"])
    scan = hdf5.root["data"][50, :, :, 4]
    mask = hdf5.root["truth"][50, :, :, 4]
    dicom_util.show_img_msk_fromarray(scan, mask, save_path="img")

def validate_npz():
    scan_file = "/home/sanjit/datasets/lung_processed/LUNG1-172/05-27-2007-StudyID-54005/scan.3d.27.npz"
    scan_data = np.load(scan_file)
    print(scan_data["scan"].shape)
    print(scan_data["mask"].shape)

#call the function to test
validate_npz()
