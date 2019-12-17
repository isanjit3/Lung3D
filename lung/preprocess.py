#Pre-process dicom files.
import os
import glob
import pydicom
import numpy as np
from numpy import savez_compressed
from unet3d import dicom_data_util
from unet3d.utils import dicom_util
import config


def pre_process_dicom_files(data_folder, process_folder, overwrite=False ):
  """
  Preprocess the dicom files. Read each of the study folder and create 3D image.
  """
  subjects = glob.glob(os.path.join(data_folder, "*"))
  
  for _, subject in enumerate(subjects):
    study_folders = glob.glob(os.path.join(subject, "*"))    
    for si, sf in enumerate(study_folders): #si - index, #sf - folder
      process_path = os.path.join(process_folder, os.path.basename(subject), os.path.basename(sf))
      if not os.path.exists(process_path):
        os.makedirs(process_path)
      elif not overwrite : 
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
        
        print('Processing Scan path: ', scan_image_path) 
        #Get the resliced image and mask data for the study.      
        img_data, mask_data = dicom_util.get_scan_and_mask_data(scan_image_path, contour_file_name, False)
      
        #Group them into 512x512x8 3D channels and store in hdf5.
        
        #start_index = 0
        #end_index = len(img_data)-1
        #slices_to_drop = len(img_data) % config.config["image_shape"][2]
        #if (slices_to_drop > 0):
        #  start_index = slices_to_drop / 2
        #  end_index = len(img_data) - slices_to_drop / 2 - 1
        
        img_count = 0
        img_temp = []
        mask_temp = []
        slice_count =0
        for i in range(0, len(img_data)):
          slice_count += 1
          img_temp.append(img_data[i])
          mask_temp.append(mask_data[i])
          
          if (slice_count == 8):
            img_temp = np.array(img_temp)
            mask_temp = np.array(mask_temp)
            img = np.dstack(img_temp[0:8])
            mask = np.dstack(mask_temp[0:8])

            #save it to the disk
            savez_compressed(os.path.join(process_path, "scan.3d." + str(img_count)), scan=img, mask=mask)
            
            img_temp = []
            mask_temp = []
            slice_count = 0
            img_count += 1
        
def write_data_file(processed_folder, overwrite=False):
  samples = glob.glob(os.path.join(processed_folder, "*", "*", "scan.*"))
  print(len(samples))
  dicom_data_util.write_data_to_file(samples, config.config["data_file"], config.config["image_shape"])
    
    #read the scan and mask file and write to hdf5 file.



#pre_process_dicom_files(config.config["data_path"], config.config["processed_data_path"])  

write_data_file(config.config["processed_data_path"])

