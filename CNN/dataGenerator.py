"""
Script containing generator for JigNet
"""
import numpy as np
import JigsawHelpers as help
import helpers as helper
import hamming
import time
import random

fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"


def generator(image_array, avail_keys, hamming_set, crop_size=25, batch_size=8, num_permutations=50):
    # Divide array into cubes
    idx_array = np.zeros((batch_size, hamming_set.shape[0]), dtype=np.uint8)
    array_list = np.zeros((batch_size, len(avail_keys)))
    while True:
        for i in range(batch_size):
            rand_idx = random.randrange(image_array.shape[0])
            random_idx = random.randrange(hamming_set.shape[0])
            # Divide image into cubes
            cells = help.divide_input(image_array[np.newaxis, rand_idx])
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
            # Shuffle according to hamming
            # Randomly assign labels to cells
            # print("Permutation:", hamming_set[random_idx])
            out_dict = help.shuffle_jigsaw(cropped_dict, hamming_set[random_idx])
            for n, val in enumerate(out_dict.values()):
                print("N:", n)
                array_list[i, n] = [helper.normalise(val)]
            idx_array[i, random_idx] = 1
        yield ({'alexnet_input_{}'.format(n): elem for n, elem in enumerate(array_list)}, {'ClassificationOutput': idx_array})


def main(num_permutations=25):
    print("Load data")
    fixed_array, fixed_affine = help.get_data(fixed_dir, moving_dir, dvf_dir)
    print("Get moveable keys")
    avail_keys = help.get_moveable_keys(fixed_array)
    # Hamming distance
    start_time = time.time()
    hamming_set = hamming.gen_max_hamming_set(num_permutations, avail_keys)
    end_time = time.time()
    print("Took {} to generate {} permutations". format(
        end_time - start_time, num_permutations))
    print("Hamming Set Shape:", hamming_set.shape)
    print(hamming_set)

    print("Generator")
    list_arrays, index, shuffle_dict, fix_dict = generator(fixed_array, avail_keys, hamming_set)
    print("Solve puzzle", index)
    puzzle_array = help.solve_jigsaw(shuffle_dict, fix_dict, fixed_array)

    helper.write_images(puzzle_array, fixed_affine,
                        file_path="./jigsaw_out/", file_prefix='rand_fix')


if __name__ == '__main__':
    main()
