# Python file containing functions required to preprocess images
import os
import pydicom



class ReadData(object):

        @staticmethod
        def read_dicom(input_path, output_filename):
            ds = pydicom.dcmread(input_path)
            with open("{}.txt".format(output_filename), 'w') as f:
                f.write(str(ds))
            return ds
