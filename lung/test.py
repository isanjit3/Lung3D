import os
import glob
import config
import numpy as np
import scipy
import random
from random import shuffle

from unet3d.utils import utils
from unet3d import dicom_data_util
from unet3d.utils import dicom_util
from unet3d import generator, model, training
from unet3d.model import isensee2017_model



def display_img_from_hdf5(img_index, slice_index):
    hdf5 = dicom_data_util.open_data_file(config.config["data_file"])
    scan = hdf5.root["data"][img_index, 0, :, :, slice_index]
    mask = hdf5.root["truth"][img_index, 0, :, :, slice_index]
    dicom_util.show_img_msk_fromarray(scan, mask, save_path="img")
    dicom_util.save_histogram(scan, save_path=os.path.abspath("histogram.png"))

def validate_npz():
    scan_file = "/home/sanjit/datasets/lung_processed/LUNG1-002/01-01-2014-StudyID-85095/scan.data.3d.0.npz"
    samples = glob.glob(os.path.join(config.config["processed_data_path"], "*", "*", "scan.*"))    
    index = random.randint(0,len(samples)-1)
    #scan_file = samples[index]

    scan_data = np.load(scan_file)
    print(scan_data["data"].shape)
    print(scan_data["truth"].shape)
    
    scan = scan_data["data"][:, :, 70]
    mask = scan_data["truth"][:, :, 70]
    
    scan = dicom_util.normalize(image = scan, min_bound = config.config["min_bound"], max_bound = config.config["max_bound"])

    dicom_util.show_img_msk_fromarray(scan, mask, save_path="img")
    dicom_util.save_histogram(scan, save_path=os.path.abspath("histogram.png"))


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

        img = scipy.ndimage.interpolation.zoom(truth_img, (0.25, 0.0025, 1))
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
                
def get_isense_model():
    model = isensee2017_model(input_shape=config.config["input_shape"], 
                                  n_labels=config.config["n_labels"],
                                  initial_learning_rate=config.config["initial_learning_rate"],
                                  n_base_filters=config.config["n_base_filters"])
    model.summary()

def get_training_validation_split():
    data_file = dicom_data_util.open_data_file(config.config["data_file"])
    nb_samples = data_file.root.data.shape[0]       
    sample_list = list(range(nb_samples))
    shuffle(sample_list)
    n_training = int(len(sample_list) * config.config["validation_split"])
    training = sample_list[:n_training]
    testing = sample_list[n_training:]
    print("Training Steps:", len(training))
    print("Validation Steps:", len(testing))

def validate_processed_lidc_files():
    skipped_subjects = []
    skipped_subjects_file = os.path.abspath("skipped_subjects.pkl")   
    if os.path.exists(skipped_subjects_file):
        skipped_subjects = utils.pickle_load(skipped_subjects_file)

    original = []
    lidc_original = glob.glob(os.path.join(config.config["datasets"], "LIDC-IDRI", "*"))
    for i in lidc_original:
        subject = os.path.basename(i)
        original.append(subject)

    processed = []
    lidc_processed = glob.glob(os.path.join(config.config["datasets"], "lidc_processed", "*"))
    for i in lidc_processed:
        subject = os.path.basename(i)
        processed.append(subject)

    not_processed = [i for i in original if i not in processed]
    not_processed = [i for i in not_processed if i not in skipped_subjects]
    not_processed_file = os.path.abspath("notprocessed.pkl")
    utils.pickle_dump(not_processed, not_processed_file)

    
#-------------------------------------------------

#display_img_from_hdf5(85, 50)

#process_single_subject("/home/sanjit/datasets/lung/LUNG1-172")
#validate_npz()

#get_training_validation_split()

validate_processed_lidc_files()

