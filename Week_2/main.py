# MPHYS project main file to control image pre-processing
import os
from readData import ReadData
from pydicom.datadict import dictionary_VR
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


def preprocess(input_path, output_filename):
    ds = readData.read_dicom(input_path)
    ds = ds.DeformableRegistrationSequence[1].DeformableRegistrationGridSequence[0].VectorGridData
    readData.write_dicom(ds, output_filename)
    print("Done")


def load_patient(dvf_path, pct_path, petct_path):
    dvf_ds, pct_ds, petct_ds = readData.load_patient_data(dvf_path, pct_path, petct_path)
    print("Done Loading Patient Info.")


def main(argv=None):
    preprocess(in_path, "dvf_data")
    #load_patient(dvf_Path, pct_Path, petct_Path)


if __name__ == '__main__':
    main()
