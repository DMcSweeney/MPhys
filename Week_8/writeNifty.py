"""
Script that loads PETCT and Planning CT and writes to .nii such that registration
can be performed.
"""
import os
import pydicom
from imageReg import ImageReg
from readMasks import ReadMasks

os.chdir('..')
cwd = os.getcwd()
print(cwd)

ImageReg = ImageReg()
ReadMasks = ReadMasks()

patient_dir = "E:\\Mphys\\Patients\\"
#pet_outdir = "E:\\Mphys\\NiftyPatients\\PET\\"
#planning_outdir = "E:\\Mphys\\NiftyPatients\\PlanningCT\\"


def load_files(patient_dir):
    """
        Function that takes the patient directory and navigates through the
        tree to find relevant files
        File paths are returned as dict with key = Patient filename
        and value = path to scan folders
    """

    pet_paths = {}
    pct_paths = {}
    struct_path = {}
    for f in os.listdir(patient_dir):
        try:
            patient_path_list = os.path.join(patient_dir, f)
            folders = [os.path.join(patient_path_list, file)
                       for file in os.listdir(patient_path_list)]
            if len(folders) == 2:
                folders.sort(key=get_size, reverse=True)
                pet_path = folders[0]  # Inside patient folder
                pct_path = folders[1]
                # Organise PET and CT folders, by number of files per folder and size
                pet_files = [os.path.join(pet_path, folder) for folder in os.listdir(pet_path)]
                pct_files = [os.path.join(pct_path, folder) for folder in os.listdir(pct_path)]
                pet_files.sort(key=lambda t: (get_number_files, get_size), reverse=True)
                pct_files.sort(key=get_number_files, reverse=True)
                struct_path[f] = [pct_files[index] for index, folder in enumerate(os.listdir(
                    pct_path)) if 'Structure' in folder]
                pet_paths[f] = pet_files[2]
                pct_paths[f] = pct_files[0]
            else:
                print("Can't extract scans")
        except IndexError:
            print("Too few files")
            pass
    return pet_paths, pct_paths, patient_path_list, struct_path


def get_number_files(start_path):
    # Function that returns number of files in directory
    # print(os.listdir(start_path))
    number_files = len([file for file in os.listdir(start_path)])
    #print("Number of files:", number_files)
    return number_files


def get_size(start_path):
    # Function that returns directory size
    total_size = 0
    for path, dirs, files in os.walk(start_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    #print("Directory size: " + str(total_size))
    return total_size


def write_nifty(pet_paths, pct_paths):
    # Function that writes image series to .nii format
    pet_series = {}
    pct_series = {}
    for key, value in pet_paths.items():
        try:
            pet_series[key] = ImageReg.load_series(value, pet_outdir, key)
            pct_series[key] = ImageReg.load_series(pct_paths[key], planning_outdir, key)
        except RuntimeError:
            print("RuntimeError Ignored")
            pass
    return pet_series, pct_series


def write_masks(patient_dir, contour_path, image_path, index, img_format='nii'):
    for folder, path in contour_path.items():
        print(contour_path[folder])
        if contour_path[folder] is not None and image_path is not None:
            try:
                ReadMasks.create_image_mask_files(
                    folder, contour_path[folder][0], image_path[folder], index, img_format)
            except PermissionError:
                print("No Permission")
                pass
        else:
            print("No Structure Set found")
    print("Done Writing masks")


def main(argv=None):
    pet_paths, pct_paths, patient_path_list, struct_path = load_files(patient_dir)
    print(struct_path)
    write_masks(patient_dir, struct_path, pct_paths, 0)
    # print(pet_paths)
    # print(len(pet_paths))
    # write_nifty(pet_paths, pct_paths)


if __name__ == '__main__':
    main()
