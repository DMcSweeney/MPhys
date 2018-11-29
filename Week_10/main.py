# MPHYS project main file to control image pre-processing
import os
import numpy as np
from readData import ReadData
from imageReg import ImageReg
from interact import Interact
import SimpleITK as sitk

# Change working directory such that it can access data
os.chdir("..")
# Print current working directory
cwd = os.getcwd()
print(cwd)
# Paths
pct_path = ".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659\\000000.dcm"
dvf_path = "E:\\Mphys\\ElastixReg\\DVF\\HN-CHUM-001\\deformationField.nii"
petct_path = ".\\Patients\\HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\\3-StandardFull-07232"
struct_path = '.\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\114120634-TomoTherapy Structure Set-68567\\000000.dcm'
ReadData = ReadData()
ImageReg = ImageReg()
Interact = Interact()


def read_dicom(dicom_path):
    # Function that reads a dicom file and writes to a text file
    dataset = ReadData.read_dicom(dicom_path)
    # vector_grid = dataset.DeformableRegistrationSequence[1].DeformableRegistrationGridSequence[0].VectorGridData
    # vector_grid = np.array(vector_grid).astype(np.float64)
    #
    # with open("dvf.raw", "wb") as f:
    #     f.write(vector_grid)
    ReadData.write_dicom(dataset, "image_test")


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
        pct_data[folder], petct_data[folder], dvf_data[folder] = ReadData.load_patient_array(
            dvf_path, pct_path, petct_path)

    return pct_data, petct_data, dvf_data


def main(argv=None):

    # read_dicom(pct_path)
    dvf = sitk.ReadImage(dvf_path)
    dvf_array = sitk.GetArrayFromImage(dvf)
    print(type(dvf_array))
    print(np.shape(dvf_array))
    print("X:", dvf_array[:, :, :, 0])
    # with open("dvf_nii.txt", 'w') as f:
    #     f.write(str(dvf_array))
    # ImageReg.load_series(
    #     ".\\Patients\\HN-CHUM-001\\08-27-1885-TomoTherapy Patient Disease-00441\\112161818-kVCT Image Set-62659\\", "pct_series")
    # ImageReg.load_series(
    #     ".\\Patients\\HN-CHUM-001\\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\\3-StandardFull-07232\\", "test_series")
    # fixed = ImageReg.image_info("pct_series.mha")
    # moving = ImageReg.image_info("petct_series.mha")
    # dvf = sitk.ReadImage("dvf.mhd", sitk.sitkVectorFloat64)
    # small_dvf = ImageReg.resize_image(dvf, fixed)
    # small_moving = ImageReg.resize_image(movsng, fixed)
    # pre_def, post_def = ImageReg.get_rigid_transforms(dvf_path)
    # transform = ImageReg.define_transform(pre_def, post_def, dvf)
    # transform_img = ImageReg.resample_image(moving, fixed, transform)
    # ImageReg.myshow(transform_img)
    # print("Done Loading Patient Info")


if __name__ == '__main__':
    main()
