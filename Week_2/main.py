# MPHYS project main file to control image pre-processing
import os
from readData import ReadData


os.chdir("D:\\Documents\GitHub\MPhys")
# Print current working directory
cwd = os.getcwd()
print(cwd)
# Paths
pct_path = "HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\
                112161818-kVCT Image Set-62659\\"
dvf_path = "HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\1-REGCTsim-CTPET-CT-43961"
petct_path = "HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\3-StandardFull-07232\\"

# Define filename within folder
dvf_filename = "000000.dcm"

# Define path to file
in_path = os.path.join(dvf_path, dvf_filename)
print(in_path)
readData = ReadData()


def preprocess(input_path, output_filename):
    ds = readData.read_dicom(input_path, output_filename)
    ds.PatientName
    print("Done")


def main(argv=None):
    preprocess(in_path, "dvf_data")


if __name__ == '__main__':
    main()
