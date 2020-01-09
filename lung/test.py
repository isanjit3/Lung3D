import os
import glob
import config
import numpy as np
import scipy
from unet3d.utils import utils
from unet3d import dicom_data_util
from unet3d.utils import dicom_util
from unet3d import generator, model, training


def display_img_from_hdf5():
    hdf5 = dicom_data_util.open_data_file(config.config["data_file"])
    scan = hdf5.root["data"][50, 0, :, :, 4]
    mask = hdf5.root["truth"][50, 0, :, :, 4]
    dicom_util.show_img_msk_fromarray(scan, mask, save_path="img")

def validate_npz():
    scan_file = "/home/sanjit/datasets/lung_processed/LUNG1-172/05-27-2007-StudyID-54005/scan.3d.27.npz"
    scan_data = np.load(scan_file)
    print(scan_data["scan"].shape)
    print(scan_data["mask"].shape)

def get_model():
    cnn_model = model.unet_model_3d(input_shape=config.config["input_shape"],
                              pool_size=config.config["pool_size"],
                              n_labels=1, #config.config["n_labels"],
                              initial_learning_rate=config.config["initial_learning_rate"],
                              deconvolution=config.config["deconvolution"],
                              depth=2 # 4 is the default.
                            )
    cnn_model.summary()


def verify_processed():
    """
    verifies if any subject folder is missed to process.
    """
    missing_subjects = []
    subjects = glob.glob(os.path.join(config.config["data_path"], "*"))
    for s in subjects:
        process_path = os.path.join(config.config["processed_data_path"], os.path.basename(s))
        if not os.path.exists(process_path):
            print(s)
            missing_subjects.append(s)
            break
        
        f = []
        for (dirpath, dirnames, filenames) in os.walk(process_path):
            f.extend(filenames)

        if (len(f) == 0):
            print(s)
            missing_subjects.append(s)
    
    return missing_subjects

def verify_min_max_values():
    samples = glob.glob(os.path.join(config.config["processed_data_path"], "*", "*", "scan.*"))      
    for s in samples:
        scan_data = np.load(s)  
        max = scan_data['data'].max()
        min = scan_data['data'].min()
        print(min, ", ", max, " :", s)



def validate_data_and_truth():
    data_file = dicom_data_util.open_data_file(config.config["data_file"])
    print(data_file.root.data.shape)
    print(data_file.root.truth.shape)
    di = 69 #data index

    truth = data_file.root.truth
    data = data_file.root.data

    truth_img = truth[di][0]
    non_zero_count = np.count_nonzero(truth_img)
    print("Non Zero Count for index: %d is %d" % (di, non_zero_count))
    if non_zero_count > 0:
        #dicom_util.save_img_3d(truth_img, os.path.abspath("channel_images.png"), 0)

        img = scipy.ndimage.interpolation.zoom(truth_img, (0.25, 1.25, 1))
        dicom_util.save_img_3d(img, save_path=os.path.abspath("rescaled_image.png"), threshold=0)
        img_x = np.zeros(truth_img.shape, truth.dtype)
        print("Scaled image shape: ", img.shape)
        print("Destination Image Shape: ", img_x.shape)
        loc = tuple(np.subtract(img_x.shape, img.shape) // 2)       
        utils.paste(img_x, img, loc)
        dicom_util.save_img_3d(img_x, save_path=os.path.abspath("resampled_image.png"), threshold=0)

        #save specific image slice
        img = truth_img[:, :, 60]
        dicom_util.save_img(img_arr = (img,), save_path=os.path.abspath("channel_images_60.png"))
                


#-------------------------------------------------
#call the function to test
#validate_npz()
#display_img_from_hdf5()
#get_model()

#verify_min_max_values()

validate_data_and_truth()
