# Python file containing functions required to preprocess images
import os
import pydicom



class ReadData(object):

        @staticmethod
        def read_dicom(input_path, output_filename):
            dataset = pydicom.dcmread(input_path)
            print("Storage type.....:", dataset.SOPClassUID)
            print()

            pat_name = dataset.PatientName
            display_name = pat_name.family_name + ", " + pat_name.given_name
            print("Patient's name...:", display_name)
            print("Patient id.......:", dataset.PatientID)
            print("Modality.........:", dataset.Modality)
            print("Study Date.......:", dataset.StudyDate)
            print("Vector Grid Data .", dataset.DeformableRegistrationSequence)
            if 'PixelData' in dataset:
                plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
                plt.show()
                rows = int(dataset.Rows)
                cols = int(dataset.Columns)
                print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                    rows=rows, cols=cols, size=len(dataset.PixelData)))
                if 'PixelSpacing' in dataset:
                    print("Pixel spacing....:", dataset.PixelSpacing)

            # use .get() if not sure the item exists, and want a default value if missing
            print("Slice location...:", dataset.get('SliceLocation', "(missing)"))
            with open("{}.txt".format(output_filename), 'w') as f:
                f.write(str(dataset))

            return dataset
