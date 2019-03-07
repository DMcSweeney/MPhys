"""
Script containing useful functions for Jigsaw CNN
"""
import numpy as np
from itertools import product
import dataLoader as load
import helpers as help
import math
import random
from numba import jit
# On server with PET and PCT in
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
    return np.mean(input_image, axis=0, keepdims=True)


def average_cell(input_dict):
    return {key: np.mean(value) for key, value in input_dict.items()}


def divide_input(input_array, number_cells_per_dim=4, dims=3):
    # Key should be cube position
    # Value is sliced array
    sliced_dims = tuple([int(x/number_cells_per_dim) for x in input_array.shape[1:4]])
    sliced_x, sliced_y, sliced_z = sliced_dims
    cells = {prod: np.array(input_array[:, prod[0]*sliced_x: prod[0]*sliced_x+sliced_x, prod[1]*sliced_y: prod[1]*sliced_y+sliced_y, prod[2]*sliced_z: prod[2]*sliced_z+sliced_z, :])
             for prod in product(range(0, number_cells_per_dim), repeat=dims)}

    return cells


def shuffle_jigsaw(input_dict, fix_dict, number_cells_per_dim=4, dims=3):
    # Randomly assign key to value
    list_keys = [key for key in input_dict.keys()]
    shuffle_keys = random.sample(list_keys, len(list_keys))
    shuffle_dict = {new_key: input_dict[shuffle_keys[i]]
                    for i, new_key in enumerate(random.sample(list_keys, len(list_keys)))}
    return shuffle_dict


def solve_jigsaw(shuffled_cells, fixed_cells, input_array):
    # Put array back together
    all_cells = {}
    puzzle_array = np.zeros(shape=input_array.shape)
    all_cells.update(shuffled_cells)
    all_cells.update(fixed_cells)
    for key, value in all_cells.items():
        x, y, z = key
        puzzle_array[:, x*value.shape[1]:x*value.shape[1]+value.shape[1], y*value.shape[2]:y *
                     value.shape[2]+value.shape[2], z*value.shape[3]: z*value.shape[3]+value.shape[3], :] = value
    return puzzle_array


def split_shuffle_fix(input_dict, threshold=-500):
    # Split into cells to shuffle and those to stay fixed
    # To reduce possible permutations
    shuffle_dict = {key: value for key, value in input_dict.items() if np.mean(value) > threshold}
    fix_dict = {key: value for key, value in input_dict.items() if np.mean(value) <= threshold}
    return shuffle_dict, fix_dict


@jit
def get_data(fixed_dir, moving_dir, dvf_dir):
    # Load data from directory
    fixed_predict, moving_predict, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)
    fixed_array, fixed_affine = fixed_predict.get_data()
    return fixed_array, fixed_affine


def main(argv=None):
    print("Load data into arrays")
    fixed_array, fixed_affine = get_data(fixed_dir, moving_dir, dvf_dir)

    print("Divide input")
    image_cells = divide_input(fixed_array)

    print("Fix cells below threshold")
    shuffle_cells, fix_cells = split_shuffle_fix(image_cells)

    print("Shuffle cells")
    shuffle_image = shuffle_jigsaw(shuffle_cells, fix_cells)

    print("{} cells have been shuffled. This is {} permutations.".format(
        len(shuffle_cells.keys()), math.factorial(len(shuffle_cells.keys()))))

<<<<<<< HEAD
def main(argv=None):
    # Load data into arrays
    fixed_array, fixed_affine = get_data(fixed_dir, moving_dir, dvf_dir)
    # Get average images'
    print("Fixed Array Shape:", fixed_array.shape)
    avg_img = average_pix(fixed_array)
    Divide input_
    fixed_cells = divide_input(avg_img)
    avg_dict = average_cell(fixed_cells)
    shuffle_image = shuffle_jigsaw(fixed_cells)
    Check shapes
    puzzle_array = solve_jigsaw(fixed_cells, fixed_array)
    print(fixed_array[1,1,1,1,0 ])
    jitter_array = jitter(fixed_array,2)
    print(jitter_array[1,1,1,1,0])
    for key, value in avg_dict.items():
        print("Avg Value in cell_{} is {}".format(key, value))
        help.write_images(puzzle_array, fixed_affine, file_path="./jigsaw_out/", file_prefix='average')
=======
    print("Solve puzzle")
    puzzle_array = solve_jigsaw(shuffle_image, fix_cells, fixed_array)

    help.write_images(puzzle_array, fixed_affine,
                      file_path="./jigsaw_out/", file_prefix='shuffle_fix')
>>>>>>> 005e29a20bc187c942bbdaf1dadb38bf83f42dcd


if __name__ == '__main__':
    main()
