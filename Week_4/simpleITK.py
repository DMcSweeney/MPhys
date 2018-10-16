from __future__ import print_function

import SimpleITK as sitk
import pydicom
import os

os.chdir('..')

data_directory = os.path.dirname(".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659\\")
series_ID = '1.3.6.1.4.1.14519.5.2.1.5168.2407.178959368858707198180439962659'

# Get the list of files belonging to a specific series ID.
reader = sitk.ImageSeriesReader()
# Use the functional interface to read the image series.
original_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, series_ID))

# Write the image.
output_file_name_3D = os.path.join('..\\MPhys', '3DImage.mha')
sitk.WriteImage(original_image, output_file_name_3D)

# Read it back again.
written_image = sitk.ReadImage(output_file_name_3D)

# Check that the original and written image are the same.
statistics_image_filter = sitk.StatisticsImageFilter()
statistics_image_filter.Execute(original_image - written_image)

# Check that the original and written files are the same
print('Max, Min differences are : {0}, {1}'.format(statistics_image_filter.GetMaximum(), statistics_image_filter.GetMinimum()))
