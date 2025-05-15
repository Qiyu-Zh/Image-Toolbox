# Image-Toolbox
```
pip install git+https://github.com/Qiyu-Zh/Image-Toolbox.git
```
Example:
```
get_contour(binary_mask):
```
return an mask image with only boundary inside.]
```
resize(image_path, target_size, output_path = None):
```
Resize a dcm file from image_path to target_size and save in output_path. If you dont want to save, you can ignore the parameter and the output will be the sitk image
```
crop_with_bbox(image_path, bbox, output_path = None):
```
Crop a dcm file and save. The bbox should be like[[2,20], [20, -1]]. Notice -1 is allowed.
```
load_2d_3d(dicom_folder, output_path = None):
```
Load all dcm images inside a folder and create a 3d volumn file.
```
plot_contour(mask_for_cv, ax, color = "red", label = None):
```
plot the contour inside the mask_for_cv to matplotlib ax. The mask should be binary.
```
plot_mask(mask_for_cv, ax, color = "Blues", alpha = 0.5, label = None):
```
plot the mask inside the mask_for_cv to matplotlib ax. The mask should be binary.
```
list_display(img_list, name="", ...)
```
Visualize corresponding slices across multiple 3D volumes side-by-side in an interactive slider-based widget, supporting DICOM and NIfTI formats.
```
folders_display(img_list, name="", ...)
```
Similar to list_display, but assumes each item is a full 3D volume and displays axial slices interactively for folder-based batch inspection.
```
plot3d(volume, mask=None, ...)
```
Plots a 3D scatter of intensity values from a volume, optionally masked and downsampled, providing quick insight into spatial distribution.
```
load_2d_3d(dicom_input, output_path=None)
```
Converts a DICOM series or list of slices into a single 3D volume, enabling standardized volumetric analysis.
```
load_3d_2d(dcm_file, output_path)
```
Slices a 3D DICOM volume into 2D NIfTI slices while preserving correct spatial metadata (origin, spacing).
```
erode3d(mask, size=2)
```
Performs 3D binary erosion on segmentation masks to refine region boundaries or isolate structures.


