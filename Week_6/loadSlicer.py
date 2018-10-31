"""
Script that loads PETCT and Planning CT into slicer and perform registration.
Returns transformed image, transform and execution time.
"""
import os
import json
import numpy as np
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


def register_img(pet_series, pct_series):
    """
    Perform registration using pyelastix library. Input image series and outputs transformed image
    and transformation field. Both written to text file.
    Output Format: image_array = str(dict{patient: arrays})
                   transform_array = str(dict{patient: arrays})

    """
    pet_image = {key: sitk.GetArrayFromImage(value) for key, value in pet_series.items()}
    pct_image = {key: sitk.GetArrayFromImage(value) for key, value in pct_series.items()}
    params = pyx.get_default_params()
    transform_img = {}
    transform_field = {}
    transform_array = {}
    for key, value in pct_image.items():
        transform_img[key], transform_field[key] = pyx.register(
            pet_image[key], value, params, exact_params=False, verbose=1)
    print("Registration Done")
    image_array = {key: np.array(value).tolist() for key, value in transform_img.items()}
    transform_array = {key: np.array(value).tolist() for key, value in transform_field.items()}
    for key, value in transform_array.items():
        with open(".\\Transform_Fields\\{}.json".format(key), "w+") as f:
            json.dump(value, f, indent=4, separators=(',', ':'))
    for key, value in image_array.items():
        with open(".\\Transform_Images\\{}.json".format(key), "w+") as f:
            json.dump(value, f, indent=4, separators=(',', ':'))
    print("Done Writing Files")
    return image_array, transform_array


def main(argv=None):
    pet_path, pct_path, patient_path_list = load_files(".\Patients")
    pet_series, pct_series = load_series(pet_path, pct_path, patient_path_list)
    transform_img, transform_field = register_img(pet_series, pct_series)
    print(type(transform_field))
    print("Done Transform")


if __name__ == '__main__':
    main()
