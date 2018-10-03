# MPHYS project main file to control image pre-processing
import os
from readData import ReadData

# Change working directory such that it can access data
os.chdir("..")
# Print current working directory
cwd = os.getcwd()
print(cwd)
# Paths
pct_Path = "HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659"
dvf_Path = "HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\1-REGCTsim-CTPET-CT-43961"
petct_Path = "HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\\3-StandardFull-07232"

# Define filename within folder
dvf_filename = "000000.dcm"

# Define path to file
in_path = os.path.join(dvf_Path, dvf_filename)
readData = ReadData()



def import_data(input_path, output_filename):
    dataset = readData.read_dicom(input_path)
    readData.write_dicom(dataset, output_filename)
    print("Done reading DICOM file. Written to text file.")
    return dataset


def load_patient(dvf_path, pct_path, petct_path):
    dvf_ds, pct_ds, petct_ds = readData.load_patient_data(dvf_path, pct_path, petct_path)
    print("Done Loading Patient Info.")
    return dvf_ds, pct_ds, petct_ds


def main(argv=None):
    # ds = import_data(in_path, "dvf_data")
    dvf_ds, pct_ds, petct_ds = load_patient(dvf_Path, pct_Path, petct_Path)
    # Extract pixel data into an pixel_array
    pct_data = readData.dataset_to_array(pct_ds)
    petct_data = readData.dataset_to_array(petct_ds)
    dvf_data = readData.load_dvf_data(dvf_ds)

    # Extract important data


if __name__ == '__main__':
    main()
