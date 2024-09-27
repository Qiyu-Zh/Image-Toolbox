import SimpleITK as sitk
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from ipywidgets import interact
from ipywidgets.widgets import IntSlider
import matplotlib.pyplot as plt
import cv2

def get_contour(binary_mask):

    # Multiply by 255 to convert it to the format expected by OpenCV (0s and 255s)
    mask_for_cv = binary_mask.astype(np.uint8) * 255
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank binary mask to draw the contours
    contour_only_mask = np.zeros_like(binary_mask).copy()
    # Draw contours on the blank mask with value 1 for the contour lines
    cv2.drawContours(contour_only_mask, contours, -1, 1, 1)  # Use 1 for binary contour and thickness 1
    # Output the contour-only mask (strictly binary, 0s and 1s)
    return contour_only_mask


def resize(image_path, target_size, output_path = None):
    image = sitk.ReadImage(image_path)
    original_size = image.GetSize()  
    original_spacing = image.GetSpacing()  
    target_size = list(target_size)
    for i in range(len(target_size)):
    	if target_size[i] == -1:
    	    target_size[i] = original_size[i]	
    new_spacing = [original_spacing[i] * (original_size[i] / target_size[i]) for i in range(len(target_size))]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitkBSpline2)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resized_image = resampler.Execute(image)
    if output_path:
        sitk.WriteImage(resized_image, out_path)
    return resized_image
  
def crop_with_bbox(image_path, bbox, output_path = None):
    image = sitk.ReadImage(image_path)
    # Extract start index and size from the bounding box
    adjusted_bbox = []
    for i in range(3):
        start = bbox[i][0] if bbox[i][0] >= 0 else image_size[i] + bbox[i][0]
        end = bbox[i][1] if bbox[i][1] >= 0 else image_size[i] + bbox[i][1]
        adjusted_bbox.append((start, end))
    bbox = adjusted_bbox
    start_index = [bbox[i][0] for i in range(3)]  # Start index (x_min, y_min, z_min)
    size = [bbox[i][1] - bbox[i][0] for i in range(3)]  # Size (x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Crop the image using RegionOfInterest
    cropped_image = sitk.RegionOfInterest(image, size=size, index=start_index)
    
    # Update the origin based on the start index
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()
    
    # Calculate the new origin based on the cropping
    new_origin = [original_origin[i] + start_index[i] * original_spacing[i] for i in range(3)]
    
    # Set the origin, direction, and spacing of the cropped image
    cropped_image.SetOrigin(new_origin)
    cropped_image.SetDirection(image.GetDirection())
    cropped_image.SetSpacing(image.GetSpacing())
    if out_path:
        sitk.WriteImage(cropped_image, output_path)
    return cropped_image
  
def load_2d_3d(dicom_folder, output_path = None):
    # Use SimpleITK to read the DICOM series
    reader = sitk.ImageSeriesReader()
    # Get the list of DICOM file names from the directory
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    # Load the DICOM series
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    if output_path:
        sitk.WriteImage(image, out_path)
    return image


  
