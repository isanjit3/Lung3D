#DICOM utility class
import os
import pydicom
import numpy as np
import scipy
import dicom_contour.contour as dcm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lung import config

import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi


#This method returns the image in the HU Units
def parse_dicom_file(fileName):
    """Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """
    try:
        dcm = pydicom.read_file(fileName)
        dcm_image = dcm.pixel_array

        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        dcm_image = dcm_image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        dcm_image[dcm_image == -2000] = 0

        try:
            intercept = dcm.RescaleIntercept        
        except AttributeError:        
            intercept = 0.0

        try:
            slope = dcm.RescaleSlope        
        except AttributeError:        
            slope = 0.0
        
        if slope != 1:
            dcm_image = slope * dcm_image.astype(np.float64)
            dcm_image = dcm_image.astype(np.int16)            
            dcm_image += np.int16(intercept)

        return np.array(dcm_image, dtype=np.int16)

    except:
        return None

def get_roi_contour_ds(rt_sequence, index):
    """
    Extract desired Regions of Interest(ROI) contour datasets
    from RT Sequence.
    
    E.g. rt_sequence can have contours for different parts of the lung
    
    You can use get_roi_names to find which index to use
    
    Inputs:
        rt_sequence (pydicom.dataset.FileDataset): Contour file dataset, what you get 
                                                after reading contour DICOM file
        index (int): Index for ROI Sequence
    Return:
        contours (list): list of ROI contour pydicom.dataset.Dataset s
    """
    if (len(rt_sequence.ROIContourSequence) == 0):
      return []

    # index 0 means that we are getting RTV information
    ROI = rt_sequence.ROIContourSequence[index]
    # get contour datasets in a list
    contours = [contour for contour in ROI.ContourSequence]
    return contours

def contour2poly(contour_dataset, path,slices_imgpath_dict):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs
        contour_dataset (pydicom.dataset.Dataset) : DICOM dataset class that is identified as
                        (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    
    if (img_ID not in slices_imgpath_dict):
      print("Image ID:", img_ID, "not found in the slice image path dict.")
      return;
    
    img = pydicom.read_file(os.path.join(path, slices_imgpath_dict[img_ID]))
    img_arr = img.pixel_array
    img_shape = img_arr.shape
    
    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((x - origin_x) / x_spacing), np.ceil((y - origin_y) / y_spacing))  for x, y, _ in coord]
    return pixel_coords, img_ID, img_shape

def poly_to_mask(polygon, width, height):
    from PIL import Image, ImageDraw
    
    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
    in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask

def get_mask_dict(contour_datasets, path, slices_imgpath_dict):
    """
    Inputs:
        contour_datasets (list): list of pydicom.dataset.Dataset for contours
        path (str): path of directory with images

    Return:
        img_contours_dict (dict): img_id : contour array pairs
    """
    
    from collections import defaultdict
    
    # create empty dict for 
    img_contours_dict = defaultdict(int)

    for cdataset in contour_datasets:
        coords, img_id, shape = contour2poly(cdataset, path, slices_imgpath_dict) or (None, None, None)
        if coords is None:
          continue
          
        mask = poly_to_mask(coords, *shape)
        img_contours_dict[img_id] += mask
    
    return img_contours_dict

def get_img_mask_voxel(slice_orders, mask_dict, image_path, slices_imgpath_dict):
    """ 
    Construct image and mask voxels
    
    Inputs:
        slice_orders (list): list of tuples of ordered img_id and z-coordinate position
        mask_dict (dict): dictionary having img_id : contour array pairs
        image_path (str): directory path containing DICOM image files
    Return: 
        img_voxel: ordered image voxel for CT/MR
        mask_voxel: ordered mask voxel for CT/MR
    """
    
    img_voxel = []
    mask_voxel = []
    tumor_only_slices = []
    for img_id, _ in slice_orders:
        path = os.path.join(image_path, slices_imgpath_dict[img_id])
        img_array = parse_dicom_file(path)
        
        if img_id in mask_dict: 
          mask_array = mask_dict[img_id]
          if (np.count_nonzero(mask_array) > 0):
            tumor_only_slices.append(1)
          else:
            tumor_only_slices.append(0)
        else: 
          mask_array = np.zeros_like(img_array)
          tumor_only_slices.append(0)
          
        img_voxel.append(img_array)
        mask_voxel.append(mask_array)
        
    return img_voxel, mask_voxel, tumor_only_slices

def show_img_msk_fromarrayX(img_arr, msk_arr, alpha=0.35, sz=7, cmap='inferno',
                           save_path=None):

    """
    Show original image and masked on top of image
    next to each other in desired size
    Inputs:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
        save_path (str): path to save the figure
    """

    msk_arr = np.ma.masked_where(msk_arr == 0, msk_arr)
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray')
    plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

def get_slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    slices_img_path = {}
    for s in os.listdir(path):
        try:
            f = pydicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
            slices_img_path[f.SOPInstanceUID] = s
        except:
            continue

    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=dcm.operator.itemgetter(1))
    """
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
      slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
      slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
      s.SliceThickness = slice_thickness
    """
    return ordered_slices, slices_img_path


def display_image(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def get_scan_data(image_path, reslice=False):
    """
    Returns the scan image when there is not contour associated with the scan.
    """
    slices = get_scan_slices(image_path)
    img_data = []
    for s in slices:
        img_data.append(s.pixel_array)

    if reslice:
        np_img = np.asarray(img_data, dtype=np.int16)
        img_data = reslice_image(np_img, slices)

    return img_data

def get_scan_and_mask_data(image_path, contour_filename, return_tumor_only_slices = False, reslice=False):
    """
    Returns the Scan and the Mask data in normalized 1x1x1 voxels
    """
    # read dataset for contour
    rt_sequence = pydicom.read_file(contour_filename)

    # get contour datasets with index idx
    idx = 0
    contour_datasets = get_roi_contour_ds(rt_sequence, idx)

    # get slice orders
    ordered_slices, slices_imgpath_dict = get_slice_order(image_path)
    
    # construct mask dictionary
    mask_dict = get_mask_dict(contour_datasets, image_path, slices_imgpath_dict)

    # get image and mask data for patient
    img_data, mask_data, tumor_only_slices = get_img_mask_voxel(ordered_slices, mask_dict, image_path, slices_imgpath_dict)
    
    #reslice the image to 1x1x1 voxel
    if reslice:
        np_img = np.asarray(img_data, dtype=np.int16)
        np_mask = np.asarray(mask_data, dtype=np.int8)

        print("Reslicing scans to 1x1x1 voxel")
        scan = get_scan_slices(image_path)
        spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
        img_data = reslice_image(np_img, spacing) 
        mask_data = reslice_image(np_mask, spacing)

    if return_tumor_only_slices and (1 in tumor_only_slices):  
        idx = tumor_only_slices.index(1)  
        ldx = max(idx for idx, val in enumerate(tumor_only_slices) if val == 1) 
        return img_data[idx:ldx], mask_data[idx:ldx] # Return only the tumor slices
    else: 
        return img_data, mask_data
 

def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names

def show_img_msk_fromarray(img_arr, msk_arr, alpha=0.35, sz=7, cmap='inferno', save_path=None):

    """
    Show original image with mask on top of image
    next to each other in desired size
    Inputs:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
        save_path (str): path to save the figure
    """
    
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray')  
    
    plt.subplot(1, 2, 2)    
    plt.imshow(msk_arr, cmap='jet', interpolation='none', alpha=None)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

def save_img(img_arr, save_path):
    """
    Saves Numpy array as image to disk.
    Inputs:
        img_arr (np.array): Numpy array containing the image data
        save_path: Path to the image where to save
    """
    plt.figure(figsize=(7, 7))
    cols = len(img_arr)
    for i in range(0, cols):
        plt.subplot(1, cols, i+1)
        plt.imshow(img_arr[i], cmap='gray') 

    plt.savefig(save_path)
    plt.close()

def reslice_image(image, spacing, new_spacing=[1,1,1]):
    """
    A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance 
    between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], 
    this can be problematic for automatic analysis (e.g. using ConvNets)!

    A common method of dealing with this is resampling the full dataset to a certain 
    isotropic resolution. If we choose to resample everything to 1mm1mm1mm pixels we 
    can use 3D convnets without worrying about learning zoom/slice thickness invariance.

    Whilst this may seem like a very simple step, it has quite some edge cases due to 
    rounding. Also, it takes quite a while.
    Inputs:
        image (np.array): 3D array containing the scan slices
        spacing: Current slice spacing
        new_spacing: required spacing between the slices.
    Returns:
        Scan slices with the new spacing.
    """
    # Determine current pixel spacing
    #spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def normalize(image, min_bound, max_bound, pixel_mean):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.

    #zero center the image so that the mean value is 0
    image = image - pixel_mean
    return image

def get_scan_slices(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) #Sort the image slices based on image position
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            
    for s in slices:
        s.SliceThickness = slice_thickness
            
    return slices

def get_scan_countour_filenames(scan_folders):
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
    
    return scan_image_path, contour_file_name

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def save_img_3d(image, save_path, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
       
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig(save_path)
    plt.close()


def get_crop_size(img, rtol=1e-8):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.        
    Returns
    -------
    slices: Start and end indexes of the cropping dimension
    """

    #img = check_niimg(img)
    #data = img.get_data()
    data = img
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)
   
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])
    
    #slices = [slice(s, e) for s, e in zip(start, end)]
    slices = [[s, e] for s, e in zip(start, end)]
    
    for s in slices:
        if (s[1]-s[0])%2 != 0:
            s[1] = min(s[1]+1, data.shape[:3][0])
    
    w_diff = (slices[0][1] - slices[0][0]) - (slices[1][1] - slices[1][0])
    h1 = abs(int(w_diff/2))
    h2 = abs(w_diff) - h1
    if w_diff > 0: #Second slice is small
        slices[1][0] = max(slices[1][0] - h1, 0)
        slices[1][1] = min(slices[1][1] + h2, data.shape[:3][0])
    elif w_diff < 0:
        slices[0][0] = max(slices[0][0] - h1, 0)
        slices[0][1] = min(slices[0][1] + h2, data.shape[:3][0])

    w_diff = (slices[0][1] - slices[0][0]) - (slices[1][1] - slices[1][0])
    if (w_diff > 0):
        slices[0][1] = slices[0][1] - w_diff
    elif w_diff < 0:
        slices[1][1] = slices[1][1] - w_diff

    #Convert to slice format
    slices = [slice(s[0], s[1]) for s in slices]
    return slices

def extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine

def slice_positions(slice_datasets):
    image_orientation = slice_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in slice_datasets]

def slice_spacing(slice_datasets):
    if len(slice_datasets) > 1:
        slice_pos = slice_positions(slice_datasets)
        slice_positions_diffs = np.diff(sorted(slice_pos))
        return np.mean(slice_positions_diffs)
    else:
        return 0.0

def get_affine_matrix(slices, slice_space=None):
    """
    Adopted from : https://github.com/innolitics/dicom-numpy/blob/master/dicom_numpy/combine_slices.py
    Inputs:
        slices: Scan image slices in the sorted order.
        slice_space: Provide the slicing space if you are going to normalize to 1x1x1 voxels 
    """
    first_dataset = slices[0] #Assuming slices are already sorted
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    if slice_space is None:
        slice_space = slice_spacing(slices)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_space

    transform[:3, 3] = first_dataset.ImagePositionPatient

    return transform