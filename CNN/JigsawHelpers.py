"""
Script containing useful functions for Jigsaw CNN
"""
import numpy as np
from itertools import product
import dataLoader as load
import helpers as help
import math
import random
# On server
fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"


class JigsawMaker:
    def __init__(self, input_array, number_cells_per_dim=4, dims=3):
        self.cell_num = number_cells_per_dim**dims
        self.cell_shape = input_array.shape[1:4]/4
        # self.max_hamming_set =  # Calc this
        self.total_permutations = math.factorial(self.cell_num)


def average_pix(input_image):
    # Average along batch axis
    return np.mean(input_image, axis=0)


def divide_input(input_array, number_cells_per_dim=4, dims=3):
    # Key should be cube position
    # Value is sliced array
    sliced_dims = tuple([int(x/number_cells_per_dim) for x in input_array.shape[1:4]])
    print("Sliced Dims:", sliced_dims)
    sliced_x, sliced_y, sliced_z = sliced_dims
    cells = {prod: np.array(input_array[:, prod[0]*sliced_x: prod[0]*sliced_x+sliced_x, prod[1]*sliced_y: prod[1]*sliced_y+sliced_y, prod[2]*sliced_z: prod[2]*sliced_z+sliced_z, :])
             for prod in product(range(0, number_cells_per_dim), repeat=dims)}
    return cells


"""
def jigsaw_mix(air_threshold=3):
    dict_of_all_input_pos = {k: v for k, v in enumerate(inputs)}
    dict_no_air = {k: v for k, v in dict_of_all_input_pos.items() if v > air_threshold else None}
    # Then ignore none when shuffling
    # Dictionary useful for putting things back together
"""


def shuffle_jigsaw(input_dict, number_cells_per_dim=4, dims=3):
    # Randomly assign key to value
    list_keys = []
    for key in input_dict.keys():
        list_keys.append(key)
    shuffle_dict = {prod: input_dict[random_key] for random_key in random.sample(
        list_keys, len(list_keys))for prod in product(range(0, number_cells_per_dim), repeat=dims)}
    return shuffle_dict


def solve_jigsaw(cells, input_array):
    puzzle_array = np.zeros(shape=input_array.shape)
    for key, value in cells.items():
        x, y, z = key
        puzzle_array[:, x*value.shape[1]:x*value.shape[1]+value.shape[1], y*value.shape[2]:y *
                     value.shape[2]+value.shape[2], z*value.shape[3]: z*value.shape[3]+value.shape[3], :] = value
    return puzzle_array


def get_data(fixed_dir, moving_dir, dvf_dir):
    print('Load data to Transform')
    fixed_predict, moving_predict, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)
    print('Turn into numpy arrays')
    fixed_array, fixed_affine = fixed_predict.get_data()
    moving_array, moving_affine = moving_predict.get_data()
    dvf_array, dvf_affine = dvf_label.get_data(is_image=False)
    return fixed_array, fixed_affine


def main(argv=None):
    # Load data into arrays
    fixed_array, fixed_affine = get_data(fixed_dir, moving_dir, dvf_dir)
    # Divide input_
    fixed_cells = divide_input(fixed_array)
    shuffle_image = shuffle_jigsaw(fixed_cells)
    # Check shapes
    puzzle_array = solve_jigsaw(shuffle_image, fixed_array)
    help.write_images(puzzle_array, fixed_affine, file_path="./jigsaw_out/", file_prefix='shuffle')


if __name__ == '__main__':
    main()
