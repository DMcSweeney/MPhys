"""
Script containing generator for JigNet
"""
import JigsawHelpers as help
import hamming
import time


def generator(image_array, batch_size=4, num_permutations=50):
    # Divide array into cubes
    cube_list, label_list = []
    while True:
        for idx in range(batch_size):
            hamming_set = {}
            cells = help.divide_input(image_array[idx,  ...])
            # jitter
            jittered_dict = {key: help.jitter(value) for key, value in cells.items()}
            # Figure out which should move
            shuffle_dict, fix_dict = help.split_shuffle_fix(jittered_dict)
            # Hamming distance
            start_time = time.time()
            hamming_set = hamming.gen_max_hamming_set(num_permutations, shuffle_dict)
            end_time = time.time()
            print("Took {} to generate {} permutations". format(
                end_time - start_time, num_permutations))
            # Shuffle according to hamming
            hamming_dict = help.shuffle_jigsaw(hamming_set)
            # Yield
            for key, value in hamming_dict.items():
                label_list.append(key)
                cube_list.append(value)
        yield ({'input': cube_list}, {'output': label_list})
