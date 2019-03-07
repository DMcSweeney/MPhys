"""
Script containing generator for JigNet
"""
import numpy as np
import JigsawHelpers as help
import hamming
import time

fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"


def generator(image_array, batch_size=1, num_permutations=50):
    # Divide array into cubes
    while True:
        for idx in range(batch_size):
            hamming_set = {}
            cells = help.divide_input(image_array[idx,  ...])
            # jitter
            jittered_dict = {key: help.jitter(value) for key, value in cells.items()}
            # Figure out which should move
            shuffle_dict, fix_dict = help.split_shuffle_fix(jittered_dict)
            input_array = np.zeros((len(shuffle_dict.keys()), cells.values(
            ).shape[0], cells.values().shape[1], cells.values().shape[2], 1))
            label_array = np.zeros((len(shuffle_dict.keys()), 1))
            # Hamming distance
            start_time = time.time()
            hamming_set = hamming.gen_max_hamming_set(num_permutations, shuffle_dict)
            end_time = time.time()
            print("Took {} to generate {} permutations". format(
                end_time - start_time, num_permutations))
            # Shuffle according to hamming
            hamming_dict = help.shuffle_jigsaw(hamming_set)
            # Yield
            for i in range(len(hamming_dict.keys())):
                for key, value in hamming_dict.items():
                    input_array[i] = value + key
                for key in shuffle_dict.keys():
                    label_array[i] = key

        print("Input Shape:", input_array.shape)
        print("Output Shape:", label_array.shape)
        return input_array, label_array
        # yield ({'input': input_array}, {'output': label_array})


def main(argv=None):
    print("Load data")
    fixed_array, fixed_affine = help.get_data(fixed_dir, moving_dir, dvf_dir)
    print("Generator")
    input_array, label_array = generator(fixed_array)
    print("Labels:", label_array)


if __name__ == '__main__':
    main()
