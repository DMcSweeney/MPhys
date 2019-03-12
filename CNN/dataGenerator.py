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
        for idx in range(batch_size):
            # Divide image into cubes
            cells = help.divide_input(image_array)
            # Jitter
            #jittered_dict = {key: help.jitter(value[idx, ...]) for key, value in cells.items()}
            # Figure out which should move
            shuffle_dict, fix_dict = help.avail_keys_shuffle(cells, avail_keys)
            # Random crop within cubes
            cropped_dict = help.random_div(shuffle_dict)
            for val in cropped_dict.values():
                print(val.shape)
            # Shuffle according to hamming
            random_idx = random.randrange(hamming_set.shape[0])
            # Randomly assign labels to cells
            print("Permutation:", hamming_set[random_idx])
            out_dict = help.shuffle_jigsaw(cropped_dict, hamming_set[random_idx])
        # Label array should be random idx

        return [val for val in out_dict.values()], random_idx, out_dict, fix_dict
        # return input_array, label_array, hamming_dict, fix_dict
        # yield ({'input': input_array}, {'output': label_array})


def main(num_permutations=25):
    print("Load data")
    fixed_array, fixed_affine = help.get_data(fixed_dir, moving_dir, dvf_dir)
    print("Get moveable keys")
    avail_keys = help.get_moveable_keys(fixed_array)
    print("Avail keys:", type(avail_keys))
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
    print("Solve puzzle")
    puzzle_array = help.solve_jigsaw(shuffle_dict, fix_dict, fixed_array)

    helper.write_images(puzzle_array, fixed_affine,
                        file_path="./jigsaw_out/", file_prefix='hamming')


if __name__ == '__main__':
    main()
