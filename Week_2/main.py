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

ReadData = ReadData()


def load_patients_array(dvf_Path, pct_Path, petct_Path):
    for folder in os.listdir('.\Patients'):
        # Need to look at structure of folder names to see if we can
        # automatically find the important folders
        pct_data = {}
        petct_data = {}
        dvf_data = {}
        dvf_Path = os.path.join('.\Patients', dvf_Path)
        pct_Path = os.path.join('.\Patients', pct_Path)
        petct_Path = os.path.join('.\Patients', petct_Path)
        print(dvf_Path)
        pct_data[folder], petct_data[folder], dvf_data[folder] = ReadData.load_patient_array(dvf_Path, pct_Path, petct_Path)
    return pct_data, petct_data, dvf_data


def main(argv=None):
    pct_data, petct_data, dvf_data = load_patients_array(dvf_Path, pct_Path, petct_Path)
    print("Done Loading Patient Info")


if __name__ == '__main__':
    main()
