import os
import shutil
import glob
import pylidc as pl
from pylidc import utils as plutils
import matplotlib.pyplot as plt
import numpy as np
from numpy import savez_compressed
from unet3d.utils import dicom_util
from unet3d.utils import utils
import skimage
import scipy
from multiprocessing import Pool
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from unet3d import dicom_data_util
import config

skipped_subjects = []
skipped_subjects_file = os.path.abspath("skipped_subjects.pkl")
"""
https://www.kaggle.com/arnavkj95/candidate-generation-and-luna16-preprocessing
"""
def get_image_and_mask(subject_id): 
    try:   
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == subject_id).first()
        scan_data = scan.to_volume()
        scan_mask = np.zeros(scan_data.shape)
        if (len(scan.annotations) > 0):
            cmask, cbbox, _ = plutils.consensus(scan.annotations)             
            scan_mask[cbbox] = cmask

        #Transpose data and mask so that it will be (z, 512, 512)
        scan_data = np.transpose(scan_data, (2, 1, 0))
        scan_mask = np.transpose(scan_mask, (2, 1, 0))

        return scan, scan_data, scan_mask
    except Exception as e:
        print("Exception occured. May be file not found:", e)
        
    return None, None, None



def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    
    #Step 1: Convert into a binary image.     
    binary = im < 604
        
    #Step 2: Remove the blobs connected to the border of the image.    
    cleared = clear_border(binary)
    
    #Step 3: Label the image.    
    label_image = label(cleared)
    
    #Step 4: Keep the labels with 2 largest areas.    
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
   
    #Step 5: Erosion operation with a disk of radius 2. This operation is 
    #seperate the lung nodules attached to the blood vessels.    
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    
    #Step 6: Closure operation with a disk of radius 10. This operation is 
    #to keep nodules attached to the lung wall.    
    selem = disk(10)
    binary = binary_closing(binary, selem)
    
    #Step 7: Fill in the small holes inside the binary mask of lungs.    
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    
    #Step 8: Superimpose the binary mask on the input image.    
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    
    return im

def segment_lung_from_ct_scan(vol):
    segmented_ct_scan = np.asarray([get_segmented_lungs(slice) for slice in vol])
    return segmented_ct_scan
    selem = ball(2)
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000
        
        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)
            
            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
    return segmented_ct_scan

def save_processed_img(img_data, mask_data, affine, save_path, subject_id = None):
    slice_count = img_data.shape[0]
    img_size = config.config["image_shape"][0]
    if slice_count < img_size:
        print("Number of slices less than ", img_size, " Skipping Subject:", subject_id)
        return
    
    blobs = slice_count // img_size
    slices_to_drop = slice_count - blobs * img_size
    initial_index = slices_to_drop // 2
    try:  
        #Construct 256x256x256 image cube
        print("Constructing 3D image cube and Resizing images to create ", config.config["image_shape"])
        for b in range(0, blobs):
            #Initialize the data and mask place holder.
            data = np.zeros(config.config["image_shape"])
            mask = np.zeros(config.config["image_shape"])

            start_index = initial_index + b * img_size
            for i in range(start_index, start_index + img_size):
                depth = i-start_index
            
                xscale = img_size/img_data[i].shape[0]
                yscale = img_size/img_data[i].shape[1]          
                data[:,:,depth] = scipy.ndimage.interpolation.zoom(img_data[i], [xscale, yscale])

                if (not (mask_data is None)):
                    mask[:,:,depth] = scipy.ndimage.interpolation.zoom(mask_data[i], [xscale, yscale])

        #save it to the disk
        print("Writing Compressed file.")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        savez_compressed(os.path.join(save_path, "scan.data.3d." + str(b)), data=data, truth=mask, affine=affine)
    except Exception as e:
        print("Failed to process the subject: ", subject_id)
        print(e)
        shutil.rmtree(save_path)

def process_subject(subject_id, overwrite = False, plot=False):
    global skipped_subjects

    if not overwrite and subject_id in skipped_subjects:
        print("Subject ID: ", subject_id, " already in the skipped list. Not processing this subject.")
        return

    background_value = 0
    tolerance=0.00001
    process_path = os.path.join(config.config["processed_data_path"], subject_id)
    if not overwrite and os.path.exists(process_path): 
      files = os.listdir(process_path)
      if (len(files) > 0):
        print("Processed 3D data already exists. Skipping Subject:", subject_id)
        return

    print("Getting data and mask for subject: ", subject_id)
    scan, data, mask = get_image_and_mask(subject_id)
    if not scan:
        return
        
    if plot:
        print("Saving original data and mask")
        dicom_util.save_img_3d(data, "lidc_data.png", threshold=604, do_transpose=True)
        dicom_util.save_img_3d(mask, "lidc_mask.png", threshold=None, do_transpose=True)

    if not (np.any(mask == 1)):   
        skipped_subjects.append(subject_id)  
        utils.pickle_dump(skipped_subjects, skipped_subjects_file)   
        print("Subject ID:", subject_id, " Does not have any mask data. Skipping process.")
        return
    #segment the lung
    segmented_lung = segment_lung_from_ct_scan(data)
    if plot:
        print("Saving segmented lung")
        dicom_util.save_img_3d(segmented_lung, "lidc_segmented_lung.png", threshold=None, do_transpose=True)
        

    #Calculate the foreground mask
    for i, image in enumerate(segmented_lung): #enumerate on scan slices on each study
      is_foreground = np.logical_or(image < (background_value - tolerance),
                                  image > (background_value + tolerance))
      if i == 0:
        foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

      foreground[is_foreground] = 1

    #Crop the image based on foreground mask
    crop_slices = dicom_util.get_crop_size(foreground) 
    slice1 = crop_slices[0]
    slice2 = crop_slices[1]

    cropped_lung_slices = data[:, slice1, slice2]
    cropped_mask = mask[:, slice1, slice2]

    if plot:
        print("Saving cropped images")
        dicom_util.save_img_3d(cropped_lung_slices, "lidc_cropped_lung_slices.png", threshold=604, do_transpose=True)
        dicom_util.save_img_3d(cropped_mask, "lidc_cropped_mask.png", threshold=None, do_transpose=True)

    cropped_lung_slices = cropped_lung_slices.astype(np.float32)
    for i in range(cropped_lung_slices.shape[0]):    
        cropped_lung_slices[i] = dicom_util.normalize(image = cropped_lung_slices[i], min_bound = 0, max_bound = 1000)
    
    # Determine current pixel spacing and reslice images to 1m3 voxels
    print("Reslicing the scan image and mask")
    spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing], dtype=np.float32)
    cropped_lung_slices = dicom_util.reslice_image(cropped_lung_slices, spacing)
    cropped_mask = dicom_util.reslice_image(cropped_mask, spacing)

    if plot:
        print("Saving resliced images")
        dicom_util.save_img_3d(cropped_lung_slices, "lidc_re-sliced_slices.png", threshold=None, do_transpose=True)
        dicom_util.save_img_3d(cropped_mask, "lidc_re-sliced_mask.png", threshold=None, do_transpose=True)

    #save the image cubes to disk
    print("Saving the processed image")
    affine = np.zeros((4,4), np.float32)
    save_processed_img(cropped_lung_slices, cropped_mask, affine, os.path.join(config.config["processed_data_path"], subject_id), "uknown")
  
def prepreocess_lidc_data(use_pool = False, overwrite=False):
    #Get all the subjects
    scans = pl.query(pl.Scan)
    subject_ids = [s.patient_id for s in scans]
    #
    print("Processing subjects")
    subject_count = 0
    if use_pool: #Use multi-threadig.
        pool = Pool(processes=8)
        result = pool.map(process_subject, subject_ids)
        pool.close()
        pool.join()
        print(result)
    else:
        for _, subject_id in enumerate(subject_ids):    
            process_subject(subject_id, overwrite=overwrite)
            subject_count += 1
            print("Processed Subject Count:", subject_count)

def write_data_file(processed_folder, overwrite=False):
  samples = glob.glob(os.path.join(processed_folder, "*", "scan.*"))
  print("Number of samples found", len(samples))

  dicom_data_util.write_data_to_file(samples, config.config["data_file"], config.config["image_shape"])

def write_merged_data_file():
    processed_folder = "/home/sanjit/datasets/lidc_processed/"
    lidc_data = glob.glob(os.path.join(processed_folder, "*", "scan.*"))
    print(len(lidc_data))

    processed_folder = "/home/sanjit/datasets/lung_processed/"
    nsclc_data = glob.glob(os.path.join(processed_folder, "*", "*", "scan.*"))
    print(len(nsclc_data))

    samples = lidc_data + nsclc_data
    print(len(samples))
    data_file = os.path.abspath("merged_lung_data_file.h5")
    dicom_data_util.write_data_to_file(samples, data_file, config.config["image_shape"])


def remove_empty_processed_dir():
    path = os.path.join(config.config["processed_data_path"], "*")
    files = glob.glob(path)
    for i, f in enumerate(files):
        files = os.listdir(f)
        if (len(files) == 0):
            shutil.rmtree(f)
    

def main(use_pool = False):
    global skipped_subjects
    #pid = "LIDC-IDRI-0008"    
    if os.path.exists(skipped_subjects_file):
        skipped_subjects = utils.pickle_load(skipped_subjects_file)

    prepreocess_lidc_data(use_pool=use_pool)
    
    #process_subject(pid, overwrite=True, plot=True)
    
    return
    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    print(len(scan.annotations))

    #convert volume to numpy array
    vol = scan.to_volume()
    vol_mask = np.zeros(vol.shape)
    print(vol.shape)

    cmask, cbbox, masks = plutils.consensus(scan.annotations)             
    vol_mask[cbbox] = cmask
    
    #vol = vol.transpose(2, 0, 1)
    #vol_mask = vol_mask.transpose(2, 0, 1)
    
    print("Segmenting the Lung...")
    segmented_lung = segment_lung_from_ct_scan(vol)
    
    print("Saving Segmented Lung...")
    dicom_util.save_img_3d(segmented_lung, "lidc_segmented_lung.png", threshold= 604, do_transpose=True)
    
    print("Saving mask Lung...")
    dicom_util.save_img_3d(vol_mask, "lidc_mask.png", threshold= None, do_transpose=True)
    
          

if __name__ == "__main__":
    #remove_empty_processed_dir()
    #main(use_pool = True)
    #write_data_file(config.config["processed_data_path"], overwrite=True)

    write_merged_data_file()