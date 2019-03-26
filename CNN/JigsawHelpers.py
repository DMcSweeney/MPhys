"""
Script containing useful functions for Jigsaw CNN
"""
import numpy as np
from itertools import product
import dataLoader as load
import helpers as help
import random
from numba import jit
# On server with PET and PCT in
fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"


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


def random_div(input_dict, crop_size=25):
    cells = {}
    for key, val in input_dict.items():
        x_rand = random.randint(0, int(val.shape[1]-crop_size))
        y_rand = random.randint(0, int(val.shape[2]-crop_size))
        z_rand = random.randint(0, int(val.shape[3]-crop_size))
        cells[key] = val[:, x_rand:x_rand+crop_size,
                         y_rand:y_rand+crop_size, z_rand:z_rand+crop_size, :]
    return cells


def shuffle_jigsaw(input_dict, hamming_set, number_cells_per_dim=4, dims=3):
    # Randomly assign key to value
    list_keys = [key for key in input_dict.keys()]
    print(len(list_keys))
    print(len(hamming_set))
    print(type(hamming_set))
    shuffle_keys = [list_keys[hamming_set[i]] for i in range(len(list_keys))]
    shuffle_dict = {key: input_dict[key] for key in shuffle_keys}
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


def split_shuffle_fix(input_dict, threshold=-700):
    # Split into cells to shuffle and those to stay fixed
    # To reduce possible permutations
    shuffle_dict = {key: value for key, value in input_dict.items() if np.mean(value) > threshold}
    fix_dict = {key: value for key, value in input_dict.items() if np.mean(value) <= threshold}
    return shuffle_dict, fix_dict


def avail_keys_shuffle(input_dict, avail_keys, threshold=-700):
    # Split into cells to shuffle and those to stay fixed
    # To reduce possible permutations
    shuffle_dict = {key: value for key, value in input_dict.items() if key in avail_keys}
    fix_dict = {key: value for key, value in input_dict.items() if key not in avail_keys}
    return shuffle_dict, fix_dict


def jitter(input_array, Jitter=2):
    # image_number = input_array.shape[0]
    print(input_array.shape)
    x_dim = input_array.shape[0] - Jitter * 2
    y_dim = input_array.shape[1] - Jitter * 2
    z_dim = input_array.shape[2] - Jitter * 2
    # return_array = np.empty((image_number, x_dim, y_dim, z_dim,1), np.float32)
    return_array = np.empty((x_dim, y_dim, z_dim, 1), np.float32)
    # for i in range(image_number):
    x_jit = random.randrange(Jitter * 2 + 1)
    y_jit = random.randrange(Jitter * 2 + 1)
    z_jit = random.randrange(Jitter * 2 + 1)
    return_array[:, :, :, :] = input_array[x_jit:x_dim +
                                           x_jit, y_jit:y_dim + y_jit, z_jit:z_dim + z_jit, :]
    return return_array


@jit
def get_data(fixed_dir, moving_dir, dvf_dir):
    # Load data from directory
    fixed, moving, dvf = load.data_reader(fixed_dir, moving_dir, dvf_dir)
    fixed_array, fixed_affine = fixed.get_data()
    return fixed_array, fixed_affine


def get_moveable_keys(input_array):
    avg_array = average_pix(input_array)
    image_cells = divide_input(avg_array)
    shuffle_cells, fix_cells = split_shuffle_fix(image_cells)
    # These are positions of cubes that can can be shuffled
    list_avail_keys = [key for key in shuffle_cells.keys()]
    return list_avail_keys


def main(argv=None):
    print("Load data into arrays")
    fixed_array, fixed_affine = get_data(fixed_dir, moving_dir, dvf_dir)

    avg_array = average_pix(fixed_array)

    print("Divide input")
    image_cells = divide_input(avg_array)

    print("Fix cells below threshold")
    shuffle_cells, fix_cells = split_shuffle_fix(image_cells)

    print("Shuffle cells")
    shuffle_image = shuffle_jigsaw(shuffle_cells, fix_cells)

    # print("{} cells have been shuffled. This is {} permutations.".format(
    #    len(shuffle_cells.keys()), math.factorial(len(shuffle_cells.keys()))))

    print("Solve puzzle")
    puzzle_array = solve_jigsaw(shuffle_image, fix_cells, fixed_array)

    help.write_images(puzzle_array, fixed_affine,
                      file_path="./jigsaw_out/", file_prefix='shuffle_fix')


if __name__ == '__main__':
    main()
