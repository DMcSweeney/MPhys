# Class that contains all simpleITK functions required
# for image resampling, as sanity checks for our code.
from __future__ import print_function
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydicom
import numpy as np
import os
import sys
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

    @staticmethod
    def resample_image(fixed_image, moving_image, dvf):
        elastix_image = sitk.ElastixImageFilter()
        elastix_image.SetFixedImage(fixed_image)
        elastix_image.SetMovingImage(moving_image)

        parameter_map = sitk.VectorOfParameterMap()
        parameter_map.append(sitk.GetDefaultParameterMap("affine"))
        parameter_map.append(sitk.GetDefaultParameterMap("bspline"))
        elastix_image.SetParameterMap(parameter_map)
        elastix_image.Execute()
        sitk.WriteImage(elastix_image.GetResultImage())

    @staticmethod
    def command_iteration(method):
        # Function to visualise progress of Optimiser
        if (method.GetOptimizerIteration() == 0):
            print("\tLevel: {0}".format(method.GetCurrentLevel()))
            print("\tScales: {0}".format(method.GetOptimizerScales()))
        print("#{0}".format(method.GetOptimizerIteration()))
        print("\tMetric Value: {0:10.5f}".format(method.GetMetricValue()))
        print("\tLearningRate: {0:10.5f}".format(method.GetOptimizerLearningRate()))
        if (method.GetOptimizerConvergenceValue() != sys.float_info.max):
            print("\tConvergence Value: {0:.5e}".format(method.GetOptimizerConvergenceValue()))

    @staticmethod
    def command_multiresolution_iteration(method):
        print("\tStop Condition: {0}".format(method.GetOptimizerStopConditionDescription()))
        print("============= Resolution Change =============")

    @staticmethod
    def deform_reg(fixed, moving, output_transform):
        initialTx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(fixed.GetDimension()))
        R = sitk.ImageRegistrationMethod()
        R.SetShrinkFactorsPerLevel([3, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 1])

        R.SetMetricAsJointHistogramMutualInformation(20)
        R.MetricUseFixedImageGradientFilterOff()
        R.MetricUseFixedImageGradientFilterOff()
        R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                        numberOfIterations=100,
                                        estimateLearningRate=R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(initialTx, inPlace=True)
        R.SetInterpolator(sitk.sitkLinear)
        R.AddCommand(sitk.sitkIterationEvent, lambda: ImageReg.command_iteration(R))
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                     lambda: ImageReg.command_multiresolution_iteration(R))

        outTx = R.Execute(fixed, moving)

        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

        displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
        displacementField.CopyInformation(fixed)
        displacementTx = sitk.DisplacementFieldTransform(displacementField)
        del displacementField
        displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,
                                                    varianceForTotalField=1.5)

        R.SetMovingInitialTransform(outTx)
        R.SetInitialTransform(displacementTx, inPlace=True)

        R.SetMetricAsANTSNeighborhoodCorrelation(4)
        R.MetricUseFixedImageGradientFilterOff()
        R.MetricUseFixedImageGradientFilterOff()

        R.SetShrinkFactorsPerLevel([3, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 1])

        R.SetOptimizerScalesFromPhysicalShift()
        R.SetOptimizerAsGradientDescent(learningRate=1,
                                        numberOfIterations=300,
                                        estimateLearningRate=R.EachIteration)

        outTx.AddTransform(R.Execute(fixed, moving))

        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

        sitk.WriteTransform(outTx,  output_transform)
