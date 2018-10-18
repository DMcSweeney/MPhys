# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our code.
from __future__ import print_function
import SimpleITK as sitk
import matplotlib as mpl
import pydicom
import numpy as np
import os
import skimage as ski
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
        print("Pixel ID:", img.GetPixelIDTypeAsString())
        return img

    @staticmethod
    def myshow(moving, title=None, margin=0.05, dpi=80):
        # Function to display image
        # Move_Array contains the transformed image we wish to analyse
        # Fix_Array contains the original that we want to compare the moving image to
        print(moving.GetOrigin())
        move_array = sitk.GetArrayFromImage(moving)
        print(np.shape(move_array))
        print(len(move_array))
        Interact.multi_slice_viewer(move_array)

    @staticmethod
    def resize_image(img, ref_img):
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(ref_img)
        out = resample.Execute(img)
        return out

    @staticmethod
    def resample_image(moving_img, fixed_img, dis):
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(fixed_img)
        dis_tx = sitk.DisplacementFieldTransform(sitk.Cast(dis, sitk.sitkVectorFloat64))
        resample.SetTransform(dis_tx)
        out = resample.Execute(moving_img)
        vis = sitk.CheckerBoard(fixed_img, out, checkerPattern=[15, 10, 1])
        sitk.WriteImage(out, "transform_img.mha")
        return vis
