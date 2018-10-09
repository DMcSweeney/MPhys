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
    def resample_image(fixed, moving):
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(fixed)
        resample.SetOutputOrigin(fixed.GetOrigin())
        resample.SetOutputSpacing(moving.GetSpacing())
        resample.SetSize(moving.GetSize())
        resample.SetInterpolator(sitk.sitkBSpline)
        resample.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*resample.GetProgress()),end=''))
        resample.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
        out = resample.Execute(moving)
        print(type(out))
        # vis = sitk.CheckerBoard(fixed, sitk.Compose([sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)]*3), checkerPattern =[15,10,1])
        return out

    @staticmethod
    def myshow(img, title=None, margin=0.05, dpi=80):
        # Function to display image
        nda = sitk.GetArrayFromImage(img)
        print(np.shape(nda))
        spacing = img.GetSpacing()
        ysize = nda.shape[1]
        xsize = nda.shape[2]
        figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
        fig = plt.figure(title, figsize=figsize, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
        extent = (0, xsize*spacing[1], 0, ysize*spacing[0])
        t = ax.imshow(nda, extent=extent, interpolation='hamming', cmap='gray')
        if(title):
            plt.title(title)
