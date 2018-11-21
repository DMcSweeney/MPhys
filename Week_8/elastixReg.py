# Script containing all SimpleElastix functions
# These perform image registration


class ElastixReg(object):

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
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetInputImage(moving)
        transformixImageFilter.SetParameterMap(outTx.GetTransformParameterMap())
        transformixImageFilter.LogToConsoleOn()
        transformixImageFilter.Execute()
        #sitk.WriteImage(transformixImageFilter.GetResultImage(), img_name)
        out_img = transformixImageFilter.GetResultImage()
        sitk.WriteTransform(outTx,  output_transform)
        return out_img

    @staticmethod
    def image_registration(fixed, moving, transform_txt, img_name):
        # Make transform
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)
        elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(transform_txt))
        elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.Execute()

        # # SimpleTransformix
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetInputImage(moving)
        transformixImageFilter.SetParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.LogToConsoleOn()
        transformixImageFilter.Execute()

        #sitk.WriteImage(transformixImageFilter.GetResultImage(), img_name)
        out_img = elastixImageFilter.GetResultImage()
        return out_img
