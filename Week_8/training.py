"""
Script to train NN
"""
import SimpleITK as sitk
import numpy as np
from imageReg import ImageReg

image_path = "D:\\Documents\\GitHub\\MPhys\\NiftyPatients\\Elastix\\Rigid\\HN-CHUM-001.nii\\result.0.mhd"
planning_path = 'D:\\Documents\\GitHub\\MPhys\\NiftyPatients\\PlanningCT\\HN-CHUM-001.nii'
start = 0
end = 100

ImageReg = ImageReg()


def slice_array(image_path, start, end):
    image = sitk.ReadImage(image_path)
    array = sitk.GetArrayFromImage(image)
    print(np.shape(array))
    # Slice array in z-axis
    sliced_array = array[start:end, :, :]
    print(np.shape(sliced_array))
    sliced_image = sitk.GetImageFromArray(sliced_array)
    sitk.WriteImage(sliced_image, "sliced_pct.mha")


def main(argv=None):
    slice_array(planning_path, start, end)
    slice_image = sitk.ReadImage("sliced_pct.mha")
    ImageReg.myshow(slice_image)


if __name__ == '__main__':
    main()
