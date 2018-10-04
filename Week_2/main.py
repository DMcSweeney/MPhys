# MPHYS project main file to control image pre-processing
import os
from readData import ReadData

# Change working directory such that it can access data
os.chdir("..")
# Print current working directory
cwd = os.getcwd()
print(cwd)
# Paths
# pct_path = "HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659"
# dvf_path = "HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\1-REGCTsim-CTPET-CT-43961"
# petct_path = "HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\\3-StandardFull-07232"

ReadData = ReadData()


def load_patients_array():
    pct_data = {}
    petct_data = {}
    dvf_data = {}
    for folder in os.listdir('.\Patients'):
        dvf_path = ReadData.find_path(folder, 'TomoTherapy Patient Disease', 'REGCTsim-CTPET-CT')
        pct_path = ReadData.find_path(folder, 'TomoTherapy Patient Disease', 'kVCT Image Set')
        petct_path = ReadData.find_path(folder, 'PANC.', 'StandardFull')
        # print(petct_path)
        pct_data[folder], petct_data[folder], dvf_data[folder] = ReadData.load_patient_array(dvf_path, pct_path, petct_path)
    return pct_data, petct_data, dvf_data


def main(argv=None):
    pct_data, petct_data, dvf_data = load_patients_array()
    ReadData.write_dicom(dvf_data, 'test')
    print("Done Loading Patient Info")


if __name__ == '__main__':
    main()
