import glob
import os
import pydicom
import numpy as np
import tables
from .normalize import normalize_data_storage, reslice_image_set

from unet3d import dicom_data_util
from unet3d.utils import dicom_util
import config

def get_study_images(study_folders, load_tumor_only_slices = True):  
  x=[]
  y=[]
  x_temp = []
  y_temp = []
  for si, sf in enumerate(study_folders):
    scan_folders = glob.glob(os.path.join(sf, "*"))
    
    if (len(scan_folders) == 2):
      scan_image_path = ''
      contour_file_name = ''
      scan = os.listdir(scan_folders[0])[0]
      file_name = os.path.join(scan_folders[0], scan)
    
      try:
        dicom_file = pydicom.read_file(file_name)     
        dicom_file.pixel_array #Check if this is scan file.
        scan_image_path = scan_folders[0]
        contour_file_name = os.path.join(scan_folders[1], os.listdir(scan_folders[1])[0])      
      except:
        #This is contour file.     
        scan_image_path = scan_folders[1]     
        contour_file_name = file_name     
      
      print('Processing Scan path: ', scan_image_path)       
      img_data, mask_data = dicom_util.get_scan_mask_data(scan_image_path, contour_file_name, load_tumor_only_slices)
      print("Scan Image length:", len(img_data))
      #Convert to numpy array for easy processing.
      np_img = np.asarray(img_data, dtype=np.float16)
      np_mask = np.asarray(mask_data, dtype=np.int8)

      np_img = dicom_util.normalize(np_img, config.MIN_BOUND, config.MAX_BOUND, config.PIXEL_MEAN)
      #Plot the tumor slices.
      #for img, mask in zip(img_data, mask_data):
      #  show_img_msk_fromarray(img, mask, sz=10, cmap='inferno', alpha=0.5)

      for i in range(0, len(np_img)):
        x_temp.append(np_img[i])
        y_temp.append(np_mask[i])
        
        if (len(x_temp) == 8):
          x_temp = np.array(x_temp)
          y_temp = np.array(y_temp)
          x.append(np.dstack(x_temp[0:8]))
          y.append(np.dstack(y_temp[0:8]))
          x_temp = []
          y_temp = []
      
      #TODO: Create 3D array of 512x512x8
      #Number of whole batches
      """
      batch_count = len(np_img)//8
      print("Batch: ", batch_count)
      for i in range(0, batch_count):
        x.append(np.dstack(np_img[i*8 : (i+1)*8]))
        y.append(np.dstack(np_mask[i*8 : (i+1)*8]))        
      """
  return np.array(x), np.array(y)

def create_data_file(out_file, n_channels, n_samples, image_shape):
  """
  Creates and initialized the hdf5 data file.
  :param outfile: Name of the hdf5 file.
  """
  hdf5_file = tables.open_file(out_file, mode='w')
  filters = tables.Filters(complevel=5, complib='blosc')
  data_shape = tuple([0, 1] + list(image_shape))
  truth_shape = tuple([0, 1] + list(image_shape))
  data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                         filters=filters, expectedrows=n_samples)
  truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                          filters=filters, expectedrows=n_samples)
  affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                          filters=filters, expectedrows=n_samples) 
  return hdf5_file, data_storage, truth_storage, affine_storage

def write_image_data_to_file(image_files, data_storage, truth_storage, affine_storage, image_shape, n_channels, 
                             truth_dtype=np.uint8, crop=True):
  fileCount = 0                             
  for scan_file in image_files:  
    print("Adding Scan file:", scan_file)  
    scan_data = np.load(scan_file)
    if (scan_data["data"].shape != image_shape):
      print("Loaded file has incorrect shape and cannot be loaded. Shape: ", scan_data['data'].shape)
      continue
    
    data_storage.append(scan_data['data'][np.newaxis][np.newaxis])   
    truth_storage.append(scan_data['truth'][np.newaxis][np.newaxis])
    #affine_storage.append(scan_data['affine'][np.newaxis])
    #Create Dummy affine
    affine = np.zeros((4,4), np.float32)
    affine_storage.append(affine[np.newaxis])

    fileCount += 1
    print("Total File Count:", fileCount)

    
  return data_storage, truth_storage, affine_storage

def add_data_to_storage(data_storage, truth_storage, subject_data, affine, n_channels, truth_dtype):
  data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
  truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
  
def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
  """
  Takes in a set of training images and writes those images to an hdf5 file.
  :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
  the same order in each tuple. The last item in each tuple must be the labeled image. 
  Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
            ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
  :param out_file: Where the hdf5 file will be written to.
  :param image_shape: Shape of the images that will be saved to the hdf5 file.
  :param truth_dtype: Default is 8-bit unsigned integer. 
  :return: Location of the hdf5 file with the image data written to it. 
  """
  n_samples = len(training_data_files)
  n_channels = 1 #len(training_data_files[0]) - 1

  try:
      hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                n_channels=n_channels,
                                                                n_samples=n_samples,
                                                                image_shape=image_shape)
  except Exception as e:
      # If something goes wrong, delete the incomplete data file
      os.remove(out_file)
      raise e

  write_image_data_to_file(training_data_files, data_storage, truth_storage, affine_storage, image_shape,
                            truth_dtype=truth_dtype, n_channels=n_channels, crop=crop)
  if subject_ids:
      hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
  if normalize:
      print("Normalizing the data storage")
      normalize_data_storage(data_storage)
      
  print("Closing the data storage")
  hdf5_file.close()
  return out_file

def open_data_file(filename, readwrite="r"):
  return tables.open_file(filename, readwrite)


