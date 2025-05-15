# Core imaging and display libraries
import SimpleITK as sitk # For medical image I/O and registration
import ants  # Advanced Normalization Tools for image registration
import nibabel as nib # For working with NIfTI format
import nibabel.processing
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

# Visualization and interactivity
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact
from ipywidgets.widgets import IntSlider
import matplotlib.pyplot as plt
import matplotlib

# General utilities
import numpy as np
import cv2
import os 
import shutil
from skimage.metrics import structural_similarity  # For SSIM calculation
from scipy import ndimage

# MONAI: Medical imaging deep learning utilities
import monai
import monai.metrics

def demons_registration(
    fixed_image, moving_image, fixed_mask=None, moving_mask=None, fixed_points=None, moving_points=None
):
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(fixed_image)
    
    initial_transform = sitk.DisplacementFieldTransform(
        transform_to_displacement_field_filter.Execute(sitk.Transform())
    )

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=2.0
    )

    registration_method.SetInitialTransform(initial_transform)

    # Set Demons metric
    registration_method.SetMetricAsDemons(10)  # intensities are equal if diff < 10HU

    # Provide masks if available
    if fixed_mask is not None:
        registration_method.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration_method.SetMetricMovingMask(moving_mask)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=20,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points are given, display similarity metric and TRE
    if fixed_points and moving_points:
        registration_method.AddCommand(
            sitk.sitkStartEvent, rc.metric_and_reference_start_plot
        )
        registration_method.AddCommand(
            sitk.sitkEndEvent, rc.metric_and_reference_end_plot
        )
        registration_method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: rc.metric_and_reference_plot_values(
                registration_method, fixed_points, moving_points
            ),
        )

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the transformation to the moving image
    registered_moving_image = sitk.Resample(
        moving_image,
        fixed_image,  # Reference space
        final_transform,
        sitk.sitkLinear,  # Interpolation method for images
        0.0,  # Default pixel value for regions outside original image
        moving_image.GetPixelID()
    )

    # Apply the transformation to the multi-label moving mask if given
    registered_moving_mask = None
    if moving_mask is not None:
        registered_moving_mask = sitk.Resample(
            moving_mask,
            fixed_image,  # Reference space
            final_transform,
            sitk.sitkNearestNeighbor,  # Nearest neighbor to preserve labels
            0,  # Background value remains 0
            moving_mask.GetPixelID()
        )

    return final_transform, registered_moving_image, registered_moving_mask
# ============================
# Orientation and NIfTI Tools
# ============================

def as_closest_canonical_nifti(path_in, path_out):
    """
    Converts a NIfTI image to the closest canonical orientation (RAS).
    
    Parameters:
        path_in (str): Input NIfTI file path.
        path_out (str): Output file path to save the reoriented image.
    """
    img_in = nib.load(path_in)
    img_out = nib.as_closest_canonical(img_in)
    nib.save(img_out, path_out)

def undo_canonical(img_can, img_orig):
    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")

    to_canonical = img_ornt  # Same as ornt_transform(img_ornt, ras_ornt)
    from_canonical = ornt_transform(ras_ornt, img_ornt)

    # Same as as_closest_canonical
    # img_canonical = img_orig.as_reoriented(to_canonical)

    return img_can.as_reoriented(from_canonical)


def undo_canonical_nifti(path_in_can, path_in_orig, path_out):
    """
    Reverts a canonicalized NIfTI image back to the original orientation.

    Parameters:
        path_in_can (str): Canonicalized NIfTI path.
        path_in_orig (str): Original NIfTI path (provides orientation reference).
        path_out (str): Output file path for reverted image.
    """
    img_can = nib.load(path_in_can)
    img_orig = nib.load(path_in_orig)
    img_out = undo_canonical(img_can, img_orig)
    nib.save(img_out, path_out)
	
def compute_hu_distance(n_classes, y_pred, y):
    """
    Computes Hausdorff distance between predicted and ground truth labels.

    Parameters:
        n_classes (int): Number of label classes (excluding background).
        y_pred (ndarray): Predicted label volume.
        y (ndarray): Ground truth label volume.

    Returns:
        List of Hausdorff distances for each class.
    """
    data = []
    for i in range(1, n_classes):
        data.append(monai.metrics.compute_hausdorff_distance((y_pred == i), (y == i), include_background=False, distance_metric='euclidean', percentile=None, directed=False, spacing=None))
    return data
    
    

def get_HU_error(y_pred, y, onehot = False, n_classes = None):
    """
    Computes Hausdorff distance with optional one-hot encoding support.

    Parameters:
        y_pred (ndarray): Predicted labels or one-hot encoded array.
        y (ndarray): Ground truth labels or one-hot encoded array.
        onehot (bool): Whether input arrays are already one-hot encoded.
        n_classes (int): Number of classes, required if onehot=False.

    Returns:
        Hausdorff distance (tensor).
    """
    def get_onehot(label, n_classes):
        one_hot = np.eye(n_classes)[label]  # Shape: (1, 32, 32, 32, 6)
        one_hot = one_hot.transpose(3, 0, 1, 2)
        one_hot = np.expand_dims(one_hot, axis=0)

        return one_hot
        
    if onehot is False:
        y = get_onehot(y, n_classes)  # Shape: (1, 32, 32, 32, 6)
        y_pred = get_onehot(y_pred, n_classes) 
    return monai.metrics.compute_hausdorff_distance(y_pred, y, include_background=False, distance_metric='euclidean', percentile=None, directed=False, spacing=None) 
	
# ============================
# Morphological Operations
# ============================    
    
def erode3d(mask=np.ones((6, 6, 6)), size=2):
    """
    Perform 3D binary erosion on a mask.

    Parameters:
        mask (ndarray): 3D binary mask.
        size (int): Radius of erosion kernel.

    Returns:
        ndarray: Eroded mask.
    """
    # Ensure mask is boolean
    mask = mask[:]
    
    # Create a 3D structuring element
    structure = np.ones((2 * size + 1, 2 * size + 1, 2 * size + 1), dtype=bool)
    
    # Perform binary erosion
    eroded_mask = ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)
    
    return eroded_mask

def plot3d(CFR_crop, mask = None, vmax = 2, sample_rate = 5):
    """
    Plot 3D scatter of volume data optionally filtered by a binary mask.

    Parameters:
        volume (ndarray): 3D scalar volume.
        mask (ndarray): Optional binary mask.
        vmax (float): Color scale maximum.
        sample_rate (int): Sampling rate for plotting.
    """
    matplotlib.use('module://ipympl.backend_nbagg')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = np.indices(CFR_crop.shape)

    # Filter the coordinates and CFR values using the mask
    if mask:
        mask = mask[:]
        mask = mask - erode3d(mask, size = 1)
        mask = mask.astype(bool)
        mask_indices = np.where(mask)

    else:
        mask_indices = CFR_crop[:].nonzero()
    x_masked = x[mask_indices]
    y_masked = y[mask_indices]
    z_masked = z[mask_indices]
    CFR_masked = CFR_crop[mask_indices]

    x_masked = x_masked[::sample_rate]
    y_masked = y_masked[::sample_rate]
    z_masked = z_masked[::sample_rate]
    CFR_masked = CFR_masked[::sample_rate]

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Interactive 3D CFR Visualization")
    # Plot only the masked values
    scatter = ax.scatter(
        x_masked,
        y_masked,
        z_masked,
        c=CFR_masked,
        cmap="jet",
        vmin=0,
        vmax=vmax,
        s=1
    )

    # Add a colorbar and adjust its position
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    colorbar.set_label("CFR Intensity")
    plt.show()
    
    
def get_contour(binary_mask):

    # Multiply by 255 to convert it to the format expected by OpenCV (0s and 255s)
    mask_for_cv = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_only_mask = np.zeros_like(binary_mask).copy()
    cv2.drawContours(contour_only_mask, contours, -1, 1, 1)  
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
    resampler.SetInterpolator(sitk.sitkBSpline2)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resized_image = resampler.Execute(image)
    if output_path:
        sitk.WriteImage(resized_image, output_path)
    return resized_image
  
def crop_with_bbox(image_path, bbox, output_path = None):
    bbox = np.array(bbox, dtype= int).tolist()
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
    if output_path:
        sitk.WriteImage(cropped_image, output_path)
    return cropped_image
  
def load_2d_3d(dicom_input, output_path=None):
    """
    Convert a folder of DICOM files or a list of DICOM file paths into a 3D NIfTI image.
    
    Parameters:
        dicom_input (str or list): Either a folder containing DICOM files or a list of file paths.
        output_path (str, optional): Path to save the output NIfTI file.

    Returns:
        sitk.Image: The 3D image.
    """
    reader = sitk.ImageSeriesReader()

    if isinstance(dicom_input, str) and os.path.isdir(dicom_input):
        # If dicom_input is a folder, get the list of DICOM files
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_input)
    elif isinstance(dicom_input, list) and all(os.path.isfile(f) for f in dicom_input):
        # If dicom_input is a list of files, use it directly
        dicom_files = dicom_input
    else:
        raise ValueError("dicom_input must be a valid folder path or a list of file paths.")

    # Load the DICOM series
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    # Save as NIfTI if output_path is provided
    if output_path:
        sitk.WriteImage(image, output_path)
        print(f"3D NIfTI saved at: {output_path}")

    return image

def plot_contour(mask_for_cv, ax, color = "red", label = None):
    contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ax.plot([], [], color = color, linewidth = 2, label = label)
    for c in contours:
        
        close_c = np.vstack([c, c[0:1]])
        ax.plot(close_c[:,0,0], close_c[:,0,1], color = color, linewidth = 2)

    return contours
    


def plot_mask(mask_for_cv, ax, color = "Blues", alpha = 0.5, label = None):
    masked_mask = np.ma.masked_where(mask_for_cv == 0, mask_for_cv)
    ax.imshow(masked_mask, alpha=alpha, cmap = color, label = None, vmin = 0, vmax=1)
    return masked_mask


def list_display(img_list, name = "", vmax = 300, cmap = 'jet', read_dcm = False):

    
    if read_dcm:
        img_list = [sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list]
    max_slices_contrast = max(img_list, key = lambda x: x.shape[2]).shape[2]

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast-1, step=1, value=0, description='Slice:')
    def display_slice(contrast_slice_index, img_list, name):
        n = len(img_list)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
        fig.suptitle(name + f'  Slice {contrast_slice_index}')
        for i, img in enumerate(img_list):
            if img is not None:
                axes[i].imshow(img[:,:,min(img.shape[2] - 1, contrast_slice_index)], cmap=cmap, vmin = 0, vmax = vmax)
                axes[i].axis('off')
        plt.show()
        
    def update(contrast_slice_index):
        display_slice(contrast_slice_index, img_list, name)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider)


def folders_display(img_list, name = "", read_dcm = False):

    if read_dcm:
        img_list = [sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list]
    max_slices_contrast = max(img_list, key = lambda x: x.shape[0]).shape[0]

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast-1, step=1, value=0, description='Slice:')

    def update(contrast_slice_index):

        display_slice(contrast_slice_index, img_list, name)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider)
def display_slice(contrast_slice_index, img_list, name):
    n = len(img_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    fig.suptitle(name + f'  Slice {contrast_slice_index}')
    for i, img in enumerate(img_list):
        if img is not None:
            axes[i].imshow(img[min(img.shape[0] - 1, contrast_slice_index),:,:], cmap='jet', vmax = 300)
            axes[i].axis('off')
    plt.show()


def lists_display(img_lists, cmap = 'jet', name = None, titles = None):
    if name == None:
        name = ""
    if titles == None:
        titles = [""] * len(img_lists)
    if isinstance(img_lists[0], str):
        img_lists = [sorted([os.path.join(folder, path) for path in os.listdir(folder) if path.endswith('.nii')]) for folder in img_lists]
        
    if isinstance(img_lists[0][0], str):
        img_lists = [[sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list] for img_list in img_lists]
    vmaxs = [300] * len(img_lists)
    for i in range(len(img_lists)):
        if np.max(img_lists[i][0]) < 50:
            vmaxs[i] = np.max(img_lists[i][0])
    max_slices_contrast = max(img_lists[0], key = lambda x: x.shape[0]).shape[0]
    max_num = len(max(img_lists, key = lambda x: len(x)))

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast-1, step=1, value=0, description='Slice:')
    list_idx_slider = widgets.IntSlider(min=0, max=max_num-1, step=1, value=0, description='img_idx:')
    def display_slice(contrast_slice_index, list_idx_slider):
        n = max(len(img_lists), 2)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
        fig.suptitle(name + f'  Slice {contrast_slice_index}')
        for i, imgs in enumerate(img_lists):
            if imgs is not None:
                
                img = imgs[min(len(imgs),list_idx_slider)]
                axes[i].imshow(img[min(img.shape[0] - 1, contrast_slice_index),:,:], cmap=cmap, vmax = vmaxs[i])
                axes[i].axis('off')
                axes[i].set_title(titles[i])
        plt.show()

    def update(list_idx, contrast_slice_index):

        display_slice(contrast_slice_index, list_idx)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider, list_idx = list_idx_slider)
    

def make_if_dont_exist(folder_path,overwrite=False):
    """
    creates a folder if it does not exists
    input: 
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder 
    """
    if os.path.exists(folder_path):
        if overwrite:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

def load_3d_2d(dcm_file, output_path):
    make_if_dont_exist(output_path)
    # Read the 3D DICOM image using SimpleITK
    image_3d = sitk.ReadImage(dcm_file)
    # Convert the 3D image to a NumPy array
    image_array_3d = sitk.GetArrayFromImage(image_3d)
    # The shape of the array is (slices, height, width)
    print(f"Shape of the 3D image array: {image_array_3d.shape}")
    spacing = image_3d.GetSpacing()
    # Loop through the slices and process each 2D slice
    for i in range(image_array_3d.shape[0]):
        # Extract 2D slice from the NumPy array
        slice_2d = image_array_3d[i, :, :]
        # Convert the 2D NumPy array back to a SimpleITK image
        slice_image = sitk.GetImageFromArray(slice_2d)
        origin = list(image_3d.GetOrigin())
        origin[2] += i * image_3d.GetSpacing()[2]  # Adjust the Z-axis position for each slice
        slice_image.SetOrigin(origin)
        slice_image.SetSpacing((spacing[0], spacing[1]))
        # Set the Instance Number (or other tags) to reflect the correct slice
        instance_number = i + 1  # Instance numbers usually start at 1
        # Save the slice as a 2D DICOM file
        sitk.WriteImage(slice_image, os.path.join(output_path, os.path.splitext(os.path.basename(dcm_file))[0] + f"_{i}.nii"))


def ssim(np_image1, np_image2):
    
    data_range = np.max([np_image1.max(), np_image2.max()]) - np.min([np_image1.min(), np_image2.min()])
    return structural_similarity(np_image1, np_image2, data_range=data_range)
    
def delete_if_exist(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been deleted.")
    else:
        print(f"File {file_path} does not exist.")
        
def sitk2ant(img, reverse = False):
    if reverse == True:
        ants.image_write(img, "tmp.nii")
        dcm = sitk.ReadImage("tmp.nii")
        delete_if_exist("tmp.nii")
        return dcm
    else:
        sitk.WriteImage(img, "tmp.nii")
        dcm = ants.image_read("tmp.nii")
        delete_if_exist("tmp.nii")
        return dcm
    
def erode(mask = np.ones((6, 6)), size = 2):
    mask = mask[:]
    structure = np.ones((2*size + 1, 2*size + 1))
    # Erode the mask
    eroded_mask = ndimage.binary_erosion(mask, structure).astype(mask.dtype)
    return eroded_mask
	
