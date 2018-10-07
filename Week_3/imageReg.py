# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our codeself.
import SimpleITK as sitk
import numpy as np
import os
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt


class ImageReg(object):

    @staticmethod
    def load_itk_image(pixel_array):
        print(pixel_array.shape())
        print("Done Loading")
