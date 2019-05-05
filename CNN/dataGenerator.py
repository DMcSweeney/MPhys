"""
Script containing generator for JigNet
"""
#from keras.utils import Sequence
import numpy as np
import JigsawHelpers as help
import helpers as helper
import random
import hamming
import pandas as pd
from keras import utils

# On server
fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"
"""
# Laptop
fixed_dir = "/mnt/e/MPhys/Data128/PlanningCT"
moving_dir = "/mnt/e/MPhys/Data128/PET_Rigid"
dvf_dir = "/mnt/e/MPhys/Data128/DVF"


fixed_dir = "D:\\Mphys\\Nifty\\PET"
moving_dir = "D:\\Mphys\\Nifty\\PCT"
dvf_dir = "D:\\Mphys\\Nifty\\DVF"
"""


def generator(image_array, avail_keys, hamming_set, img_idx=None, hamming_idx=None, crop_size=28, batch_size=8, N=25):
    # Divide array into cubes
    while True:
        idx_array = np.zeros((batch_size, hamming_set.shape[0]), dtype=np.uint8)
        array_list = np.zeros((batch_size, len(avail_keys), crop_size, crop_size, crop_size, 1))
        for i in range(batch_size):
            # rand_idx = random image
            if img_idx is None:
                rand_idx = random.randrange(image_array.shape[0])
            else:
                rand_idx = int(random.choice(img_idx))

            # random_idx = random permutation
            if hamming_idx is None:
                random_idx = random.randrange(hamming_set.shape[0])
            else:
                random_idx = int(random.choice(hamming_idx))
            # Divide image into cubes
            cells = help.divide_input(image_array[np.newaxis, rand_idx])
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
            # Shuffle according to hamming
            # Randomly assign labels to cells
            # dummy_dict = helper.dummy_dict(cropped_dict)
            out_dict = help.shuffle_jigsaw(cropped_dict, hamming_set.loc[random_idx, :].values)

            for n, val in enumerate(out_dict.values()):
                array_list[i, n, ...] = val
            idx_array[i, random_idx] = 1

        inputs = [array_list[:, n, ...] for n in range(len(avail_keys))]
        yield inputs, idx_array
        # return inputs, idx_array, random_idx_list, rand_idx_list


def predict_generator(image_array, avail_keys, hamming_set, hamming_idx=None, image_idx=None, blank_idx=None, crop_size=28, batch_size=8, N=25):
    # Divide array into cubes
    while True:
        array_list = np.zeros((batch_size, len(avail_keys), crop_size, crop_size, crop_size, 1))
        for i in range(batch_size):
            # rand_idx = random image
            if image_idx is None:
                rand_idx = random.randrange(image_array.shape[0])
            else:
                rand_idx = int(image_idx[i])
            # random_idx = random permutation
            if hamming_idx is None:
                random_idx = random.randrange(hamming_set.shape[0])
            else:
                random_idx = int(hamming_idx[i])
            # Divide image into cubes
            cells = help.divide_input(image_array[np.newaxis, rand_idx])
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Blank out cubes = to blank_idx in avail_keys

            if blank_idx is not None:
                for idx in blank_idx:
                    blank_key = avail_keys[idx]
                    shuffle_dict[blank_key] = np.zeros(shape=(1, 32, 32, 32, 1))
            else:
                pass
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
            # Shuffle according to hamming
            # Randomly assign labels to cells
            # print("Permutation:", hamming_set[random_idx])
            # dummy_dict = helper.dummy_dict(cropped_dict)
            out_dict = help.shuffle_jigsaw(cropped_dict, hamming_set.loc[random_idx, :].values)
            for n, val in enumerate(out_dict.values()):
                array_list[i, n, ...] = val
        # return array_list, idx_array, out_dict, fix_dict
        inputs = [array_list[:, n, ...] for n in range(len(avail_keys))]
        yield inputs


def evaluate_generator(image_array, avail_keys, hamming_set, hamming_idx=None, image_idx=None, blank_idx=None, out_crop=False, inner_crop=False, crop_size=28, batch_size=8, N=25):
    # Divide array into cubes
    while True:
        idx_array = np.zeros((batch_size, hamming_set.shape[0]), dtype=np.uint8)
        array_list = np.zeros((batch_size, len(avail_keys), crop_size, crop_size, crop_size, 1))
        for i in range(batch_size):
            # rand_idx = random image
            if image_idx is None:
                rand_idx = random.randrange(image_array.shape[0])
            else:
                rand_idx = int(image_idx[i])
            # random_idx = random permutation
            if hamming_idx is None:
                random_idx = random.randrange(hamming_set.shape[0])
            else:
                random_idx = int(hamming_idx[i])
            # Divide image into cubes
            cells = help.divide_input(image_array[np.newaxis, rand_idx])
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Blank out cubes = to blank_idx in avail_keys
            if blank_idx is not None:
                for idx in blank_idx:
                    blank_key = avail_keys[idx]
                    shuffle_dict[blank_key] = np.zeros(shape=(1, 32, 32, 32, 1))
            else:
                pass
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
            fix_dict = help.random_div(fix_dict)
            if out_crop is True:
                cropped_dict = help.outer_crop(cropped_dict)
            else:
                pass
            if inner_crop is True:
                cropped_dict = help.inner_crop(cropped_dict)
            else:
                pass
            # Shuffle according to hamming
            # Randomly assign labels to cells
            # print("Permutation:", hamming_set[random_idx])
            # dummy_dict = helper.dummy_dict(cropped_dict)
            out_dict = help.shuffle_jigsaw(cropped_dict, hamming_set.loc[random_idx, :].values)
            for n, val in enumerate(out_dict.values()):
                array_list[i, n, ...] = val
            idx_array[i, random_idx] = 1
        # return array_list, idx_array, out_dict, fix_dict
        inputs = [array_list[:, n, ...] for n in range(len(avail_keys))]
        # yield inputs, idx_array
        return out_dict, fix_dict


"""
class mygenerator(Sequence):
    def __init__(self, image_set, batch_size):
        self.x = image_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
"""


def main(N=10, batch_size=2):
    print("Load data")
    fixed_array, fixed_affine = help.get_data(fixed_dir, moving_dir, dvf_dir)
    #normalised_dataset = helper.normalise(test_dataset)
    print("Get moveable keys")
    avail_keys = pd.read_csv("avail_keys_both.txt", sep=",", header=None)
    list_avail_keys = [(avail_keys.loc[i, 0], avail_keys.loc[i, 1], avail_keys.loc[i, 2])
                       for i in range(len(avail_keys))]
    # Get hamming set
    print("Load hamming Set")
    hamming_set = pd.read_csv(
        "mixed_hamming_set.txt", sep=",", header=None)
    hamming_set = hamming_set.loc[:99]
    hamming_idx = [12]
    img_idx = [0]

    shuffle_dict, fix_dict = evaluate_generator(
        fixed_array, list_avail_keys, hamming_set, hamming_idx=hamming_idx, image_idx=img_idx, blank_idx=None, out_crop=True, inner_crop=False, batch_size=1, N=10)

    # cropped_fixed = help.random_div(fix_dict)
    puzzle_array = help.solve_jigsaw(shuffle_dict, fix_dict, fixed_array)
    helper.write_images(puzzle_array, fixed_affine,
                        file_path="./oclusion_test/", file_prefix='out_crop')


if __name__ == '__main__':
    main()
