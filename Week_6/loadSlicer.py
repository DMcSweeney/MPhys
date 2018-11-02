"""
Script that loads PETCT and Planning CT into slicer and perform registration.
Returns transformed image, transform and execution time.
"""
import os
try:
    import simplejson as json
except ImportError:
    import json
import numpy as np
import SimpleITK as sitk
import pyelastix as pyx
from imageReg import ImageReg
import timeit
import matplotlib.pyplot as plt

os.chdir('..')
cwd = os.getcwd()
print(cwd)

ImageReg = ImageReg()


def load_files(patient_dir):
    """
        Function that takes the patient directory and navigates through the
        tree to find relevant files
        File paths are returned as dict with key = Patient filename
        and value = path to scan folders
    """
    pet_paths = {}
    pct_paths = {}
    for f in os.listdir(patient_dir):
        patient_path_list = os.path.join(patient_dir, f)
        folders = [os.path.join(patient_path_list, file) for file in os.listdir(patient_path_list)]
        folders.sort(key=get_size, reverse=True)
        pet_path = folders[0]  # Inside patient folder
        pct_path = folders[1]
        # Organise PET and CT folders, by number of files per folder and size
        pet_files = [os.path.join(pet_path, folder) for folder in os.listdir(pet_path)]
        pct_files = [os.path.join(pct_path, folder) for folder in os.listdir(pct_path)]
        pet_files.sort(key=lambda t: (get_number_files, get_size), reverse=True)
        pct_files.sort(key=get_number_files, reverse=True)
        pet_paths[f] = pet_files[2]
        pct_paths[f] = pct_files[0]
    return pet_paths, pct_paths, patient_path_list


def get_number_files(start_path):
    # print(os.listdir(start_path))
    number_files = len([file for file in os.listdir(start_path)])
    print("Number of files:", number_files)
    return number_files


def get_size(start_path):
    total_size = 0
    for path, dirs, files in os.walk(start_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    print("Directory size: " + str(total_size))
    return total_size


def load_series(pet_paths, pct_paths):
    pet_series = {}
    pct_series = {}
    for key, value in pet_paths.items():
        pet_series[key] = ImageReg.load_series_no_write(value)
        pct_series[key] = ImageReg.load_series_no_write(pct_paths[key])
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
    print(type(pet_image["HN-CHUM-001"]))
    print(np.shape(pet_image["HN-CHUM-001"]))
    rigid_params = pyx.get_default_params(type='RIGID')
    spline_params = pyx.get_default_params(type='BSPLINE')
    params = rigid_params + spline_params
    print(params)
    transform_img = {}
    transform_field = {}
    transform_array = {}
    reg_time = {}
    for key, value in pct_image.items():
        toc = timeit.default_timer()
        transform_img[key], transform_field[key] = pyx.register(
            pet_image[key], value, params, exact_params=False, verbose=1)
        tic = timeit.default_timer()
        reg_time[key] = tic - toc
    print("Registration Done")
    image_array = {key: np.array(value).tolist() for key, value in transform_img.items()}
    transform_array = {key: np.array(value).tolist() for key, value in transform_field.items()}
    show_image(image_array["HN-CHUM-001"])
    for key, value in transform_array.items():
        with open(".\\Transform_Fields\\{}.json".format(key), "w+") as f:
            json.dump(value, f, indent=4, separators=(',', ':'))
    for key, value in image_array.items():
        with open(".\\Transform_Images\\{}.json".format(key), "w+") as f:
            json.dump(value, f, indent=4, separators=(',', ':'))
    with open("registration_file.json", "w+") as f:
        json.dump(reg_time, f, indent=4, separators=(',', ':'))
    print("Done Writing Files")
    return image_array, transform_array


def pretransformed_pet(pet_series, pct_series):
    transformed_pet_series = {}
    for key, petvalue in pet_series.items():
        transformed_pet_series[key] = ImageReg.initial_transform(pct_series[key], petvalue)
    return transformed_pet_series


def read_reg(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


def write_data():
    pet_paths, pct_paths, patient_path_list = load_files(".\Patients")
    print("Found Scans")
    pet_series, pct_series = load_series(pet_paths, pct_paths)
    print("Image Series Read")
    transformed_pet_series = pretransformed_pet(pet_series, pct_series)
    transform_img, transform_field = register_img(pet_series, pct_series)
    return transform_img, transform_field


def show_image(image_array):
    data = np.asarray(image_array).astype(np.float64)
    image = sitk.GetImageFromArray(data)
    ImageReg.myshow(image)
    sitk.WriteImage(image, "imagetest.mha")


def main(argv=None):
    transform_img, transform_field = write_data()
    #data = read_reg(".\\Transform_Images\\HN-CHUM-001.json")
    # show_image(data)


if __name__ == '__main__':
    main()
