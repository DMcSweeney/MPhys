# Python file containing functions required to preprocess images
import os
import pydicom


class ReadData(object):

        @staticmethod
        def read_dicom(input_path):
            dataset = pydicom.dcmread(input_path, force=True)
            print("Storage type.....:", dataset.SOPClassUID)
            print()

            pat_name = dataset.PatientName
            display_name = pat_name.family_name + ", " + pat_name.given_name
            print("Patient's name...:", display_name)
            print("Patient id.......:", dataset.PatientID)
            print("Modality.........:", dataset.Modality)
            print("Study Date.......:", dataset.StudyDate)
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
            return dataset

        @staticmethod
        def write_dicom(dataset,  output_filename):
            with open("{}.txt".format(output_filename), 'w') as f:
                f.write(str(dataset))

        @staticmethod
        def load_patient_data(dvf_path, pct_path, petct_path):
            # Load dvf
            for filename in os.listdir(dvf_path):
                dvf_file = filename
            dvf_file_path = os.path.join(dvf_path, dvf_file)
            print(dvf_file_path)
            dataset_dvf = pydicom.dcmread(dvf_file_path)
            # Load PetCt directory
            pct_dict = {}
            for filename in os.listdir(pct_path):
                pct_file_path = os.path.join(pct_path, filename)
                print(pct_file_path)
                pct_dict[filename] = pydicom.dcmread(pct_file_path)
            # Load PCt directory
            petct_dict = {}
            for filename in os.listdir(petct_path):
                petct_file_path = os.path.join(petct_path, filename)
                print(petct_file_path)
                petct_dict[filename] = pydicom.dcmread(petct_file_path)

            return dataset_dvf, pct_dict, petct_dict

        @staticmethod
        def load_dvf_data(dataset):
            dvf_data = dataset.DeformableRegistrationSequence[1].DeformableRegistrationGridSequence[0].VectorGridData
            return dvf_data

        @staticmethod
        def dataset_to_array(dataset):
            pixel_data = {}
            for key, value in dataset.items():
                pixel_data[key] = value.pixel_array
            return pixel_data
