# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our code.
from __future__ import print_function
import SimpleITK as sitk
import matplotlib as mpl
import pydicom
import numpy as np
import os
import math
from readData import ReadData
from interact import Interact

mpl.rc('image', aspect='equal')
Interact = Interact()
ReadData = ReadData()


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
    def load_series_no_write(data_directory):
        # Function to read dicom series and write to .mha file
        for filename in os.listdir(data_directory):
            data_path = os.path.join(data_directory, filename)
            ds = pydicom.dcmread(data_path)
            series_ID = ds.SeriesInstanceUID
        # Get the list of files belonging to a specific series IDself
        reader = sitk.ImageSeriesReader()
        # Use the functional interface to read the image series.
        original_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, series_ID))
        return original_image

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
    def get_rigid_transforms(reg_path):
        dataset = ReadData.read_dicom(reg_path)
        # Pre-Deformation
        pre_def = dataset.DeformableRegistrationSequence[
            1].PreDeformationMatrixRegistrationSequence[0].FrameOfReferenceTransformationMatrix
        pre_def = np.array(pre_def).astype(np.float64)
        pre_def = np.reshape(pre_def, (4, 4))
        # Post-Deformation
        post_def = dataset.DeformableRegistrationSequence[
            1].PostDeformationMatrixRegistrationSequence[0].FrameOfReferenceTransformationMatrix
        post_def = np.array(post_def).astype(np.float64)
        post_def = np.reshape(post_def, (4, 4))
        return pre_def, post_def

    @staticmethod
    def define_transform(pre_def, post_def, dis):
        # Pre-rigid
        pre_rigid_center = dis.GetOrigin()
        print(pre_rigid_center)
        pre_theta_x = math.acos(pre_def[0, 0])
        pre_theta_y = math.acos(pre_def[1, 1])
        pre_theta_z = math.acos(pre_def[2, 2])
        pre_translation = list(pre_def[:3, 3]*10)
        print(pre_translation)
        pre_rigid = sitk.Euler3DTransform(
            pre_rigid_center, pre_theta_x, pre_theta_y, pre_theta_z, pre_translation)
        # Non-rigid
        dis_tx = sitk.DisplacementFieldTransform(sitk.Cast(dis, sitk.sitkVectorFloat64))
        # Post-rigid
        post_rigid_center = dis.GetOrigin()
        post_theta_x = math.acos(post_def[0, 0])
        post_theta_y = math.acos(post_def[1, 1])
        post_theta_z = math.acos(post_def[2, 2])
        post_translation = list(post_def[: 3, 3]*10)
        post_rigid = sitk.Euler3DTransform(
            post_rigid_center, post_theta_x, post_theta_y, post_theta_z, post_translation)

        composite_transform = sitk.Transform(post_rigid)
        composite_transform.AddTransform(dis_tx)
        composite_transform.AddTransform(pre_rigid)
        sitk.WriteTransform(composite_transform, "composite.tfm")
        return transform

    @staticmethod
    def resample_image(moving_img, fixed_img, transform):
        # Function that performs transform on PETCT
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(moving_img)
        resample.SetTransform(transform)
        out = resample.Execute(fixed_img)
        vis = sitk.CheckerBoard(fixed_img, out, checkerPattern=[15, 10, 1])
        sitk.WriteImage(out, "transform_img.mha")
        return out



    @staticmethod
    def initial_transform(referenceImage, floating):
         #rferenceImage = sitk.ReadImage(referenceImage ,sitk.sitkFloat32)
        # floating = sitk.ReadImage(floating, sitk.sitkFloat32)
         ## Start rigid registration
         initial_transform = sitk.CenteredTransformInitializer(referenceImage,floating,sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.GEOMETRY)
         floating_resampled = sitk.Resample(floating, referenceImage, initial_transform, sitk.sitkLinear, 0.0, referenceImage.GetPixelID())

        ## Now do proper affine registration
         registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
         registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
    #     registration_method.SetMetricAsCorrelation()
         registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
         registration_method.SetMetricSamplingPercentage(0.05)
        # registration_method.SetMetricFixedMask(referenceMask)
        # registration_method.SetMetricMovingMask(floatingMask)

         registration_method.SetInterpolator(sitk.sitkLinear)
         # Optimizer settings.
         registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
         registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
         registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
         registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
         registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
         registration_method.SetInitialTransform(initial_transform, inPlace=False)

         final_transform = registration_method.Execute(sitk.Cast(referenceImage, sitk.sitkFloat32), sitk.Cast(floating, sitk.sitkFloat32))


         floating_resampled_affine = sitk.Resample(floating, referenceImage, final_transform, sitk.sitkBSpline, 0.0, referenceImage.GetPixelID())


         #sitk.WriteImage(floating_resampled_affine,'floating_resampled_affine.mhd')
         return floating_resampled_affine
