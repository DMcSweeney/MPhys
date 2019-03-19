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


def generator(image_array, avail_keys, hamming_set, batch_size=1, num_permutations=50):
    # Divide array into cubes
    while True:
        for i in range(batch_size):
            # Divide image into cubes
            cells = help.divide_input(image_array)
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
            # Shuffle according to hamming
            random_idx = random.randrange(hamming_set.shape[0])
            # Randomly assign labels to cells
            print("Permutation:", hamming_set[random_idx])
            out_dict = help.shuffle_jigsaw(cropped_dict, hamming_set[random_idx])
            array_list = [helper.normalise(val) for val in out_dict.values()]
        yield ({'alexnet_input_{}'.format(n+1): np.array(elem) for n, elem in enumerate(array_list)}, {'ClassificationOutput': random_idx})


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
