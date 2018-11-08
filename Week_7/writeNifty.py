"""
Script that loads PETCT and Planning CT and writes to .nii such that registration
can be performed.
"""
import os
from imageReg import ImageReg
from optparse import OptionParser
from pathlib import PureWindowsPath
os.chdir('..')
cwd = os.getcwd()
print(cwd)

ImageReg = ImageReg()
parser = OptionParser()

parser.add_option("--patient_dir", dest="patient_dir",
                  help="Directory containing all patient folders", metavar="DIRECTORY")
parser.add_option("--pet_outdir", dest="pet_outdir",
                  help="Directory .nii pet scan should be written to", metavar="DIRECTORY")
parser.add_option("--planning_outdir", dest="planning_outdir",
                  help="Directory .nii planning ct should be written to", metavar="DIRECTORY")
(opt, args) = parser.parse_args()

patient_dir = PureWindowsPath(opt.patient_dir)
pet_outdir = PureWindowsPath(opt.pet_outdir)
planning_outdir = PureWindowsPath(opt.planning_outdir)


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
    # Function that returns number of files in directory
    # print(os.listdir(start_path))
    number_files = len([file for file in os.listdir(start_path)])
    print("Number of files:", number_files)
    return number_files


def get_size(start_path):
    # Function that returns directory size
    total_size = 0
    for path, dirs, files in os.walk(start_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    print("Directory size: " + str(total_size))
    return total_size


def write_nifty(pet_paths, pct_paths):
    # Function that writes image series to .nii format
    pet_series = {}
    pct_series = {}
    for key, value in pet_paths.items():
        pet_series[key] = ImageReg.load_series(value, 'patient_{}'.format(key))
        pct_series[key] = ImageReg.load_series(
            pct_paths[key], 'patient_{}'.format(key))
    return pet_series, pct_series


def main(argv=None):
    try:
        pet_paths, pct_paths, patient_path_list = load_files(patient_dir)
    except FileNotFoundError:
        print("Inconsistent file format")
    write_nifty(pet_paths, pct_paths)


if __name__ == '__main__':
    main()
