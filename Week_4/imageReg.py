# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our code.
from __future__ import print_function
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydicom
import numpy as np
import os
from interact import Interact

mpl.rc('image', aspect='equal')
Interact = Interact()


class ImageReg(object):

    @staticmethod
    def load_series(data_directory, output_filename):
        # Function to read dicom series and write to .mha file
        for filename in os.listdir(data_directory):
            data_path = os.path.join(data_directory, filename)
            ds = pydicom.dcmread(data_path)
            series_ID = ds.SeriesInstanceUID
        # Get the list of files belonging to a specific series IDself
        reader = sitk.ImageSeriesReader()
        # Use the functional interface to read the image series.
        original_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, series_ID))
        # Write the image.
        output_filename = os.path.join('..\\MPhys\\', '{}.mha'.format(output_filename))
        sitk.WriteImage(original_image, output_filename)

    @staticmethod
    def image_info(img_series):
        # Function to read .mha file into a sitk.Image format
        img = sitk.ReadImage(img_series, sitk.sitkFloat32)
        print("Size:", img.GetSize())
        print("Origin:", img.GetOrigin())
        print("Pixel Spacing:", img.GetSpacing())
        print("Direction:", img.GetDirection())
        return img

    @staticmethod
    def resample_image(fixed_image, moving_image, dvf):
        # Takes fixed image, applies dvf should return moving imag
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # moving_resampled = sitk.Resample(
        #     moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        # resample = sitk.ResampleImageFilter()
        # resample.SetReferenceImage(fixed_image)
        # # resample.SetSize(fixed.GetSize())
        # # resample.SetOutputDirection(fixed.GetDirection())
        # # resample.SetOutputSpacing(dvf.GetSpacing())
        # # resample.SetOutputOrigin(fixed.GetOrigin())
        dis_tx = sitk.DisplacementFieldTransform(sitk.Cast(dvf, sitk.sitkVectorFloat64))
        inv_dis_tx = dis_tx.GetInverse()
        # resample.SetTransform(inv_dis_tx)
        # out = resample.Execute(moving_resampled)
        out = sitk.Resample(moving_image, fixed_image, inv_dis_tx,
                            sitk.sitkNearestNeighbor, 0.0, fixed_image.GetPixelIDValue())
        return out

    @staticmethod
    def myshow(moving, fixed, title=None, margin=0.05, dpi=80):
        # Function to display image
        # Move_Array contains the transformed image we wish to analyse
        # Fix_Array contains the original that we want to compare the moving image to
        print(moving.GetOrigin())
        move_array = sitk.GetArrayFromImage(moving)
        fix_array = sitk.GetArrayFromImage(fixed)
        print(np.shape(move_array))
        print(len(move_array))
        Interact.multi_slice_viewer(move_array)
