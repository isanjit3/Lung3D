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
  for si, sf in enumerate(study_folders): #si - index, #sf - folder
    process_path = os.path.join(process_folder, os.path.basename(subject), os.path.basename(sf))
    if not os.path.exists(process_path):
      os.makedirs(process_path)
    elif not overwrite : 
      print("Processed 3D data already exists. Skipping:", sf)
      continue

    scan_folders = glob.glob(os.path.join(sf, "*"))
    scan_image_path, contour_file_name = get_scan_countour_filenames(scan_folders)

    print('Processing Scan path: ', scan_image_path) 
    #Get the resliced image and mask data for the study.   
    if (not (contour_file_name is None)):
      img_data, mask_data = dicom_util.get_scan_and_mask_data(scan_image_path, contour_file_name, False)
    else:
      img_data = dicom_util.get_scan_data(scan_image_path)
  
    slice_count = len(img_data)
    img_size = config.config["image_shape"][0]
    if slice_count < img_size:
      print("Number of slices less than ", img_size, " Skipping:", scan_image_path)
      continue
    
    data = np.zeros(config.config["image_shape"])
    mask = np.zeros(config.config["image_shape"])
    slices_to_drop = slice_count - img_size
    start_index = int(slices_to_drop/2)
      
    #Construct 256x256x256 image cube
    print("Constructing 3D image cube and Resizing images to create ", config.config["image_shape"])
    for i in range(start_index, start_index+img_size):
      depth = i-start_index
      #TODO: Crop image based on Cropping parameters
      xscale = img_size/img_data[i].shape[0]
      yscale = img_size/img_data[i].shape[1]          
      data[:,:,depth] = scipy.ndimage.interpolation.zoom(img_data[i], [xscale, yscale])

      if (not (contour_file_name is None)):
        mask[:,:,depth] = scipy.ndimage.interpolation.zoom(mask_data[i], [xscale, yscale])
        
    #save it to the disk
    print("Writing Compressed file.")
    savez_compressed(os.path.join(process_path, "scan.data.3d"), data=data, truth=mask)


def get_complete_foreground(subjects, background_value=0, tolerance=0.00001, save_segmented_lung=False):
  for subject_idx, subject in enumerate(subjects):   #Enumerate on subjects
    
    study_folders = glob.glob(os.path.join(subject, "*"))    
    for si, sf in enumerate(study_folders):  #enumerate on studies
      scan_folders = glob.glob(os.path.join(sf, "*"))
      scan_image_path, contour_file_name = dicom_util.get_scan_countour_filenames(scan_folders)

      scan_slices = dicom_util.get_scan_slices(scan_image_path)
      slices = np.stack([s.pixel_array for s in scan_slices])
      slices[slices == -2000] = 0
      
      #Get the resliced image and mask data for the study.         
      if (not (contour_file_name is None)):
        img_data, mask_data = dicom_util.get_scan_and_mask_data(scan_image_path, contour_file_name, 
                                                              return_tumor_only_slices=False, reslice=False)
        mask_data = np.array(mask_data)
      else:
        img_data = np.array(slices)
        mask_data = np.zeros(img_data.shape)
      
      #Get the Segmented Lung Slices
      segmented_lung_slices = dicom_util.segment_lung_mask(slices)

      if save_segmented_lung:
        print("Saving segmented 3D lung as image at: ", config.config['segmented_lung_img'])
        dicom_util.save_img_3d(image=segmented_lung_slices, save_path=config.config['segmented_lung_img'], threshold=0)
        dicom_util.save_img_3d(image=mask_data, save_path=config.config['segmented_mask_img'], threshold=0)

      #Calculate the foreground mask
      for i, image in enumerate(segmented_lung_slices): #enumerate on scan slices on each study
        is_foreground = np.logical_or(image < (background_value - tolerance),
                                    image > (background_value + tolerance))
        if i == 0:
          foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1

      if subject_idx == 0:
          subject_foreground = foreground
      else:
          subject_foreground[foreground > 0] = 1

    #Crop the image based on foreground mask
    crop_slices = dicom_util.get_crop_size(subject_foreground) 
    slice1 = crop_slices[0]
    slice2 = crop_slices[1]

    cropped_lung_slices = segmented_lung_slices[:, slice1, slice2]
    cropped_mask = mask_data[:, slice1, slice2]
    if save_segmented_lung:
      print("Saving Cropped 3D lung as image at: ", config.config['cropped_lung_img'])
      dicom_util.save_img_3d(image=cropped_lung_slices, save_path=config.config['cropped_lung_img'], threshold=0)
      dicom_util.save_img_3d(image=cropped_mask, save_path=config.config['cropped_mask_img'], threshold=0)

    break

  return subject_foreground


def pre_process_dicom_files(data_folder, process_folder, overwrite=False ):
  """
  Preprocess the dicom files. Read each of the study folder and create 3D image.
  """
  subjects = glob.glob(os.path.join(data_folder, "*"))
  print("Total Subjects found:", len(subjects))

  #TODO: Get the cropping parameters across all the subjects
  foreground_mask = get_complete_foreground(subjects = subjects, save_segmented_lung=False)
  savez_compressed(config.config["mask_file"], mask=foreground_mask)
  dicom_util.save_img(img_arr=(foreground_mask,), save_path=config.config["mask_file_img"])
  return

  subject_count = 0
  
  if True: #Use multi-threadig.
    pool = Pool(processes=8)
    pool.map(partial(process_dicom, process_folder=process_folder, overwrite=overwrite ), subjects)
  else:
    for _, subject in enumerate(subjects):    
      process_dicom(subject, process_folder, overwrite=overwrite)
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


pre_process_dicom_files(config.config["data_path"], config.config["processed_data_path"])  

#Create hdf5 file
#write_data_file(config.config["processed_data_path"])


#subject_folder = os.path.join(config.config["data_path"], "LUNG1-362")
#process_dicom(subject = subject_folder, process_folder = config.config["processed_data_path"], overwrite=True)

#normalize_data()