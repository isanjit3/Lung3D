#Pre-process dicom files.
import os
import glob
import pydicom
import numpy as np
import scipy
from numpy import savez_compressed
from multiprocessing import Pool
from functools import partial
from unet3d import dicom_data_util
from unet3d.utils import dicom_util
from unet3d.utils.nilearn_custom_utils.nilearn_utils import crop_img
import config

def process_dicom(subject, process_folder, overwrite=False):
  """
  Process the folder for one subject.
  """
  study_folders = glob.glob(os.path.join(subject, "*"))    
  for _, sf in enumerate(study_folders): #si - index, #sf - folder
    process_path = os.path.join(process_folder, os.path.basename(subject), os.path.basename(sf))
    if not os.path.exists(process_path):
      os.makedirs(process_path)
    elif not overwrite : 
      print("Processed 3D data already exists. Skipping:", sf)
      continue

    scan_folders = glob.glob(os.path.join(sf, "*"))
    scan_image_path, contour_file_name = dicom_util.get_scan_countour_filenames(scan_folders)

    print('Processing Scan path: ', scan_image_path) 
    #Get the resliced image and mask data for the study.   
    if (not (contour_file_name is None)):
      img_data, mask_data = dicom_util.get_scan_and_mask_data(scan_image_path, contour_file_name, False)
    else:
      img_data = dicom_util.get_scan_data(scan_image_path)
      mask_data = None
  
    scan_slices = dicom_util.get_scan_slices(scan_image_path)
    affine = dicom_util.get_affine_matrix(scan_slices)
    save_processed_img(img_data, mask_data, affine, process_path,  scan_image_path=scan_image_path)

def save_processed_img(img_data, mask_data, affine, save_path, scan_image_path = None):
  slice_count = img_data.shape[0]
  img_size = config.config["image_shape"][0]
  if slice_count < img_size:
    print("Number of slices less than ", img_size, " Skipping:", scan_image_path)
    return
  
  blobs = slice_count // img_size
  slices_to_drop = slice_count - blobs * img_size
  start_index = slices_to_drop // 2
    
  #Construct 256x256x256 image cube
  print("Constructing 3D image cube and Resizing images to create ", config.config["image_shape"])
  for b in range(0, blobs):
    #Initialize the data and mask place holder.
    data = np.zeros(config.config["image_shape"])
    mask = np.zeros(config.config["image_shape"])

    start_index = start_index + b * img_size
    for i in range(start_index, start_index + img_size):
      depth = i-start_index
      #TODO: Crop image based on Cropping parameters
      xscale = img_size/img_data[i].shape[0]
      yscale = img_size/img_data[i].shape[1]          
      data[:,:,depth] = scipy.ndimage.interpolation.zoom(img_data[i], [xscale, yscale])

      if (not (mask_data is None)):
        mask[:,:,depth] = scipy.ndimage.interpolation.zoom(mask_data[i], [xscale, yscale])
        
    #save it to the disk
    print("Writing Compressed file.")
    savez_compressed(os.path.join(save_path, "scan.data.3d." + str(b)), data=data, truth=mask, affine=affine)

def process_subject_folder(subject, overwrite=False):
  """
  Pre-process the given subject folder. Gets the scan image slices and the mask
  calculates the foreground image across all the slices and crop the slices to match the foreground
  once cropped, it reslices the image and masks to 1mm3 voxels. Then saves the blobs of images to disk.  
  """
  print("Processing Subject:", subject)
  process_folder = config.config["processed_data_path"]
  save_segmented_lung = False
  background_value = 0
  tolerance=0.00001

  study_folders = glob.glob(os.path.join(subject, "*"))    
  for _, sf in enumerate(study_folders):  #enumerate on studies
    process_path = os.path.join(process_folder, os.path.basename(subject), os.path.basename(sf))
    if not os.path.exists(process_path):
      os.makedirs(process_path)
    elif not overwrite : 
      print("Processed 3D data already exists. Skipping:", sf)
      return

    scan_folders = glob.glob(os.path.join(sf, "*"))
    is_scan_img_dir = dicom_util.is_scan_image_directory(scan_folders[0])
    if is_scan_img_dir:
      scan_image_path = scan_folders[0]
    else:
      contour_file_path = scan_folders[0]

  scan_image_path, contour_file_name = dicom_util.get_scan_countour_filenames(scan_folders)

  scan_slices = dicom_util.get_scan_slices(scan_image_path)
  slices = np.stack([s.pixel_array for s in scan_slices])
  slices = slices.astype(np.int16)
  if slices.min() == 0:
    slices = slices-1024

  slices[slices == -2000] = 0

  #Get the resliced image and mask data for the study. 
  has_mask = True        
  if (not (contour_file_name is None)):
    img_data, mask_data = dicom_util.get_scan_and_mask_data(scan_image_path, contour_file_name, 
                                                          return_tumor_only_slices=False, reslice=False)
    img_data = np.asarray(img_data)
    mask_data = np.array(mask_data)
  else:
    img_data = np.array(slices)
    mask_data = np.zeros(img_data.shape)
    has_mask = False

  if img_data.min() == 0:
    img_data = img_data-1024
  
  #Get the affine matrix. You are going to normalize to 1x1x1 voxels.
  affine = dicom_util.get_affine_matrix(scan_slices, 1.0)

  #Get the Segmented Lung Slices
  print("Segmenting the Lung")
  segmented_lung_slices = dicom_util.segment_lung_mask(slices)

  if save_segmented_lung:
    print("Saving segmented 3D lung as image at: ", config.config['segmented_lung_img'])
    dicom_util.save_img_3d(image=segmented_lung_slices, save_path=config.config['segmented_lung_img'], threshold=0)
    if has_mask:
      dicom_util.save_img_3d(image=mask_data, save_path=config.config['segmented_mask_img'], threshold=0)

  #Calculate the foreground mask
  for i, image in enumerate(segmented_lung_slices): #enumerate on scan slices on each study
    is_foreground = np.logical_or(image < (background_value - tolerance),
                                image > (background_value + tolerance))
    if i == 0:
      foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

    foreground[is_foreground] = 1

    #dicom_util.save_img((foreground,), config.config["mask_file_img"])
  #Crop the image based on foreground mask
  crop_slices = dicom_util.get_crop_size(foreground) 
  slice1 = crop_slices[0]
  slice2 = crop_slices[1]

  cropped_lung_slices = img_data[:, slice1, slice2]
  cropped_mask = mask_data[:, slice1, slice2]
  if save_segmented_lung:
    print("Saving Cropped 3D lung as image at: ", config.config['cropped_lung_img'])
    dicom_util.save_img_3d(image=cropped_lung_slices, save_path=config.config['cropped_lung_img'], threshold=0)
    if has_mask:
      dicom_util.save_img_3d(image=cropped_mask, save_path=config.config['cropped_mask_img'], threshold=0)

  # Determine current pixel spacing and reslice images to 1m3 voxels
  print("Reslicing the scan image and mask")
  spacing = np.array([scan_slices[0].SliceThickness] + list(scan_slices[0].PixelSpacing), dtype=np.float32)
  cropped_lung_slices = dicom_util.reslice_image(cropped_lung_slices, spacing)
  cropped_mask = dicom_util.reslice_image(cropped_mask, spacing)

  #save the image cubes to disk
  print("Saving the processed image")
  save_processed_img(cropped_lung_slices, cropped_mask, affine, process_path, scan_image_path)
  


def pre_process_dicom_files(data_folder, process_folder, overwrite=False, use_pool=False ):
  """
  Preprocess the dicom files. Read each of the study folder and create 3D image.
  """
  subjects = glob.glob(os.path.join(data_folder, "*"))
  print("Total Subjects found:", len(subjects))

  #TODO: Get the cropping parameters across all the subjects
  #foreground_mask = get_complete_foreground(subjects = subjects, save_segmented_lung=False)
  #savez_compressed(config.config["mask_file"], mask=foreground_mask)
  #dicom_util.save_img(img_arr=(foreground_mask,), save_path=config.config["mask_file_img"])
  #return

 
  subject_count = 0
  if use_pool: #Use multi-threadig.
    pool = Pool(processes=8)
    result = pool.map(process_subject_folder, subjects)
    pool.close()
    pool.join()
    print(result)
  else:
    for _, subject in enumerate(subjects):    
      process_subject_folder(subject, overwrite=overwrite)
      subject_count += 1
      print("Processed Subject Count:", subject_count)          


def write_data_file(processed_folder, overwrite=False):
  samples = glob.glob(os.path.join(processed_folder, "*", "*", "scan.*"))
  print(len(samples))
  dicom_data_util.write_data_to_file(samples, config.config["data_file"], config.config["image_shape"])
    
    #read the scan and mask file and write to hdf5 file.

def normalize_data():
  hdf5 = dicom_data_util.open_data_file(config.config["data_file"])
  ds = hdf5.root["data"]
  dicom_data_util.normalize_data_storage(ds)

#===============================================================================

pre_process_dicom_files(data_folder = "/home/sanjit/datasets/LIDC-IDRI", 
                        process_folder ="/home/sanjit/datasets/lidc-idri-processed", 
                        overwrite=True, 
                        use_pool=False
                      )  


#Create hdf5 file
#write_data_file(config.config["processed_data_path"])

#subject_folder = os.path.join(config.config["data_path"], "LUNG1-362")
#process_dicom(subject = subject_folder, process_folder = config.config["processed_data_path"], overwrite=True)

#normalize_data()