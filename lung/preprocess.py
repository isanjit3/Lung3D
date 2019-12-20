#Pre-process dicom files.
import os
import glob
import pydicom
import numpy as np
import scipy
from numpy import savez_compressed
from unet3d import dicom_data_util
from unet3d.utils import dicom_util
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

    if (len(scan_folders) == 2): # we expect only two folders under each study
      scan_image_path = ''
      contour_file_name = ''
      scan = os.listdir(scan_folders[0])[0]
      file_name = os.path.join(scan_folders[0], scan)
    
      try:
        dicom_file = pydicom.read_file(file_name)     
        dicom_file.pixel_array #Check if this is scan file. If not scan file, this will throw exception
        scan_image_path = scan_folders[0]
        contour_file_name = os.path.join(scan_folders[1], os.listdir(scan_folders[1])[0])      
      except:
        #This is contour file.     
        scan_image_path = scan_folders[1]     
        contour_file_name = file_name     
    elif (len(scan_folders) == 1):
      scan_image_path = scan_folders[0]
      contour_file_name = None

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
      xscale = img_size/img_data[i].shape[0]
      yscale = img_size/img_data[i].shape[1]          
      data[:,:,depth] = scipy.ndimage.interpolation.zoom(img_data[i], [xscale, yscale])

      if (not (contour_file_name is None)):
        mask[:,:,depth] = scipy.ndimage.interpolation.zoom(mask_data[i], [xscale, yscale])
        
    #save it to the disk
    print("Writing Compressed file.")
    savez_compressed(os.path.join(process_path, "scan.data.3d"), data=data, truth=mask)

def pre_process_dicom_files(data_folder, process_folder, overwrite=False ):
  """
  Preprocess the dicom files. Read each of the study folder and create 3D image.
  """
  subjects = glob.glob(os.path.join(data_folder, "*"))
  print("Total Subjects found:", len(subjects))
  subject_count = 0
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
write_data_file(config.config["processed_data_path"])


#subject_folder = os.path.join(config.config["data_path"], "LUNG1-362")
#process_dicom(subject = subject_folder, process_folder = config.config["processed_data_path"], overwrite=True)

#normalize_data()