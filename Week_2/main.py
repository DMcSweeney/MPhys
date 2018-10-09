# MPHYS project main file to control image pre-processing
import os
from readData import ReadData
from imageReg import ImageReg
# Change working directory such that it can access data
os.chdir("..")
# Print current working directory
cwd = os.getcwd()
print(cwd)
# Paths
pct_path = ".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659"
dvf_path = ".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\1-REGCTsim-CTPET-CT-43961"
petct_path = ".\\Patients\\HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\\3-StandardFull-07232"
# Read series variables
data_directory = os.path.dirname(".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659\\")
series_ID = '1.3.6.1.4.1.14519.5.2.1.5168.2407.178959368858707198180439962659'

ReadData = ReadData()
ImageReg = ImageReg()


def read_dicom(dicom_path):
    # Function that reads a dicom file and writes to a text file
    ds = ReadData.read_dicom(dicom_path)
    ReadData.write_dicom(ds, "pct_dicom")


def load_patients_array():
    # Function that loads all important data for every patient
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
    ImageReg.load_series(".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659\\", "pct_series")
    ImageReg.load_series(".\\Patients\\HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\\3-StandardFull-07232\\", "petct_series")
    fixed = ImageReg.image_info("pct_series.mha")
    moving = ImageReg.image_info("petct_series.mha")
    img = ImageReg.resample_image(fixed, moving)
    print("Size:", img.GetSize())
    print("Origin:", img.GetOrigin())
    print("Pixel Spacing:", img.GetSpacing())
    print("Direction:", img.GetDirection())
    # print(vis.shape())
    ImageReg.myshow(img)
    print("Done Loading Patient Info")


if __name__ == '__main__':
    main()
