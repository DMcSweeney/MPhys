# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our codeself.
from __future__ import print_function
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import pydicom
import numpy as np

mpl.rc('image', aspect = 'equal')


class ImageReg(object):

    @staticmethod
    def load_itk_image(pixel_array):
        img = sitk.GetImageFromArray(pixel_array)
        img = sitk.Cast(img, sitk.sitkFloat32)
        pixel_array = sitk.GetArrayFromImage(img)
        print(img.GetSize())
        print(img.GetDirection())
        print(img.GetPixelIDTypeAsString())
        plt.imshow(pixel_array)
        plt.show()
        print("Done Loading Image")

    @staticmethod
    def load_series(data_directory, output_filename):
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
        img = sitk.ReadImage(img_series)
        print("Size:", img.GetSize())
        print("Origin:", img.GetOrigin())
        print("Pixel Spacing:", img.GetSpacing())
        print("Direction:", img.GetDirection())
        return img

    @staticmethod
    def resample_image(fixed, moving, dvf):
        # Takes fixed image, applies dvf should return moving image
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(moving)
        # resample.SetSize(fixed.GetSize())
        # resample.SetOutputDirection(dvf.GetDirection())
        # resample.SetOutputSpacing(dvf.GetSpacing())
        # resample.SetOutputOrigin(fixed.GetOrigin())
        dis_tx = sitk.DisplacementFieldTransform(sitk.Cast(dvf, sitk.sitkVectorFloat64))
        resample.SetTransform(dis_tx)
        out = resample.Execute(moving)
        return out

    @staticmethod
    def myshow(moving, fixed, title=None, margin=0.05, dpi=80):
        # Function to display image
        move_array = sitk.GetArrayFromImage(moving)
        fix_array = sitk.GetArrayFromImage(fixed)
        print(np.shape(move_array))
        print(len(move_array))
        for i in range(len(move_array)):
            plt.imshow(fix_array[i], interpolation='spline16', extent=[0,1000,0,1000])
            plt.imshow(move_array[i], alpha=0.5, cmap=plt.cm.spring, interpolation='spline16', extent=[0,1000,0,1000])
            plt.pause(0.01)
            plt.draw()
