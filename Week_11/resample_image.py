import SimpleITK as sitk
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--input_image", dest="input_image", help="Path to input image", metavar="FILE")
parser.add_option("--output_image", dest="output_image",
                  help="Path to output image", metavar="FILE")
(opt, args) = parser.parse_args()


# Just something I had lying around
inputImage = opt.input_image
outputImage = opt.output_image
# Load the image. Must be a simpleITK image for this stage to work!
startImage = sitk.ReadImage(inputImage)

# factors to downsize by, this will take a 512x512x118 image to 128x128x118
factors = [4, 4, 1]
# NB, if you wanted to resample to a uniform number of slices, you could figure out what the 1 at the end should be


# Calculate the new pixel spacing and image size
new_spacing = [a*b for a, b in zip(startImage.GetSpacing(), factors)]
new_size = [int(a/b) for a, b in zip(startImage.GetSize(), factors)]

# do some in-plane blurring
# this comes from the Nyquist-Shannon theorem: https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem
sig = 2.0/(2.0*np.sqrt(np.pi))
# Blur in each plane separately (NB; there may be a more elegant way to do this, but I didn't try)
blurredImage = sitk.RecursiveGaussian(startImage, sigma=sig*new_spacing[0], direction=0)
blurredImage = sitk.RecursiveGaussian(blurredImage, sigma=sig*new_spacing[1], direction=1)

# Build the resampling filter, and then apply it
resampleFilter = sitk.ResampleImageFilter()
downsampledImage = resampleFilter.Execute(blurredImage,  # thing to resample
                                          new_size,  # size to resample to
                                          sitk.Transform(),  # How to transform - in this case, identity since we're just resampling on a coarser grid
                                          sitk.sitkBSpline,  # could also do sitk.sitkLinear which is faster, but worse
                                          startImage.GetOrigin(),  # Keep the same origin
                                          new_spacing,  # The new pixel size
                                          startImage.GetDirection(),  # Keep the same slice direction
                                          0,  # Default pixel value
                                          startImage.GetPixelIDValue())  # output pixel type

# Write the result to check
sitk.WriteImage(downsampledImage, outputImage)
