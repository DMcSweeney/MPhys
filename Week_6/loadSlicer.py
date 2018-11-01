"""
Script that loads PETCT and Planning CT into slicer and perform registration.
Returns transformed image, transform and execution time.
"""
import os
import json
import SimpleITK as sitk
import pyelastix as pyx
from imageReg import ImageReg

os.chdir('..')
cwd = os.getcwd()
print(cwd)

ImageReg = ImageReg()


def load_files(patient_dir, reverse=False):
    patient_path_list = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir)]
    folders = [os.path.join(patient, file)
               for patient in patient_path_list for file in os.listdir(patient)]
    pet_folders = folders[::2]
    pct_folders = folders[1::2]
    pet_path = [os.path.join(path, folder) for path in pet_folders for folder in os.listdir(
        path) if 'StandardFull' in folder]
    pct_path = [os.path.join(path, folder) for path in pct_folders for folder in os.listdir(
        path) if 'kVCT Image Set' in folder]
    return pet_path, pct_path, patient_path_list


def load_series(pet_path, pct_path, patient_path_list):
    patient_list = [os.path.split(path)[1] for path in patient_path_list]
    pet_series = {}
    pct_series = {}
    for i, patient in enumerate(patient_list):
        pet_series[patient] = ImageReg.load_series_no_write(pet_path[i])
        pct_series[patient] = ImageReg.load_series_no_write(pct_path[i])
    return pet_series, pct_series

def pretransformed_pet(pet_series, pct_series):
    transformed_pet_series = {}
    for key, petvalue in pet_series.items():
            print(type(petvalue))
            transformed_pet_series[key] = ImageReg.initial_transform(pct_series[key] ,petvalue)
    return transformed_pet_series

def register_img(pet_series, pct_series):
    pet_image = {key: sitk.GetArrayFromImage(value) for key, value in pet_series.items()}
    pct_image = {key: sitk.GetArrayFromImage(value) for key, value in pct_series.items()}
    params = pyx.get_default_params()
    transform = {key: pyx.register(pet_image[key], fixed, params, exact_params=False, verbose=2)
                 for key, fixed in pct_image.items()}
    with open("transform_data.txt", "w") as f:
        f.write(json.dumps(transform))
    return transform


def main(argv=None):
    pet_path, pct_path, patient_path_list = load_files(".\\Patients")
    pet_series, pct_series = load_series(pet_path, pct_path, patient_path_list)
    transformed_pet_series = pretransformed_pet(pet_series, pct_series)
    transform = register_img(transformed_pet_series, pct_series)
    print("Done Transform")


if __name__ == '__main__':
    main()
