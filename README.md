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
interactive_display(img_list, name = ""):
```
Visualization for all 3d images in img_list by slice. Notice all image should have the same number of slices. Name is the title
