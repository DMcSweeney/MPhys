# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our codeself.
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', aspect ='equal')


class ImageReg(object):

    @staticmethod
    def load_itk_image(pixel_array):
        img = sitk.GetImageFromArray(pixel_array)
        img = sitk.Cast(img, sitk.sitkFloat32)
        print(img.GetSize())
        plt.imshow(pixel_array)
        plt.show()
        print("Done Loading Image")
