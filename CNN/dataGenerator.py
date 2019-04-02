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


def generator(image_array, avail_keys, hamming_set, hamming_idx=None, crop_size=25, batch_size=8, N=25):
    # Divide array into cubes
    rand_idx_list = []
    random_idx_list = []
    while True:
        idx_array = np.zeros((batch_size, hamming_set.shape[0]), dtype=np.uint8)
        array_list = np.zeros((batch_size, len(avail_keys), crop_size, crop_size, crop_size, 1))
        for i in range(batch_size):
            # rand_idx = random image
            rand_idx = random.randrange(image_array.shape[0])
            rand_idx_list.append(rand_idx)
            # random_idx = random permutation
            if hamming_idx is None:
                random_idx = random.randrange(hamming_set.shape[0])
            else:
                random_idx = int(hamming_idx[i])

            random_idx_list.append(random_idx)
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
        # yield inputs, idx_array
        return inputs, idx_array, random_idx_list, rand_idx_list


def predict_generator(image_array, avail_keys, hamming_set, hamming_idx=None, crop_size=25, batch_size=8, N=25):
    # Divide array into cubes
    while True:
        idx_array = np.zeros((batch_size, hamming_set.shape[0]), dtype=np.uint8)
        array_list = np.zeros((batch_size, len(avail_keys), crop_size, crop_size, crop_size, 1))
        for i in range(batch_size):
            # rand_idx = random image
            rand_idx = random.randrange(image_array.shape[0])
            # random_idx = random permutation
            if hamming_idx is None:
                random_idx = random.randrange(hamming_set.shape[0])
            else:
                random_idx = int(hamming_idx[i])
            print(random_idx)
            # Divide image into cubes
            cells = help.divide_input(image_array[np.newaxis, rand_idx])
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
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
        print("Len In:", len(inputs))
        yield inputs, idx_array


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
    print("Get moveable keys")
    avail_keys = help.get_moveable_keys(fixed_array)
    # max_dist_set, dist_array = hamming.gen_max_hamming_set(N, avail_keys)
    # np.savetxt("hamming_set.txt", max_dist_set, delimiter=",", fmt='%1.2i')
    hamming_set = pd.read_csv("hamming_set.txt", sep=",", header=None)
    print("Hamming Set Shape:", hamming_set.shape)
    print(hamming_set)
    print("Generator")
    # list_arrays, index_array, shuffle_dict, fix_dict = generator(
    #    fixed_array, avail_keys, hamming_set, batch_size=2, N=10)
    inputs, idx_array, random_idx_list, rand_idx_list = generator(
        fixed_array, avail_keys, hamming_set, batch_size=32, N=10)
    print("\n\nImage:", rand_idx_list)
    print("\n\nPerm:", random_idx_list)

    np.savetxt("image_idx.txt", rand_idx_list, delimiter=",", fmt='%1.2i')
    np.savetxt("perm_idx.txt", random_idx_list, delimiter=",", fmt='%1.2i')

    """
    # cropped_fixed = help.random_div(fix_dict)
    print("Solve puzzle number:", index_array)
    # puzzle_array = help.solve_jigsaw(shuffle_dict, cropped_fixed, fixed_array)
    dummy_list = []
    for n in range(batch_size):
        check_dummy = [np.mean(list_arrays[n, i, ...]) for i in range(len(avail_keys))]
        dummy_list.append(check_dummy)
    print(dummy_list)
    # helper.write_images(puzzle_array, fixed_affine,
    #                    file_path="./jigsaw_out/", file_prefix='no_padding')

    """


if __name__ == '__main__':
    main()
