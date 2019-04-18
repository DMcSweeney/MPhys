"""
Script containing functions necessary for hamming calculations
"""
import itertools
import numpy as np
import random
import time
#from numba import jit
import pandas as pd


def HIS_gen_max_hamming_set(N, moving_cells):
    """
    Generate permutation set with max hamming distance
    N - number of permutations in returned set
    """
    # Figure out number of moving and fixed cells
    # moving_cells = [key for key in moving_dict.keys()]
    num_moving = len(moving_cells)
    # Sample 1 M permutations since 64! is too large (~10^89)
    NUM_PERMUTATIONS = 1000000
    # permutation set contains 1M permutations of all moving elements
    permutation_set = np.zeros((NUM_PERMUTATIONS, num_moving), dtype=np.uint8)
    # Populate this array
    for i,  elem in enumerate(list(itertools.islice(itertools.permutations(range(num_moving), num_moving), NUM_PERMUTATIONS))):
        permutation_set[i, ...] = elem
    # Array containing the permutations with top permutation dist.
    max_hamming_set = np.zeros((N, num_moving), dtype=np.uint8)
    j = random.randint(0, NUM_PERMUTATIONS)
    hamming_dist = np.zeros((N, NUM_PERMUTATIONS), dtype=np.uint8)

    for i in range(N):
        a = time.time()
        # Randomly assign a value to each element of max_hamming_set
        # Here, we move one element from permutation_set into max_hamming_set
        max_hamming_set[i] = permutation_set[j]
        # Replace the jth element by shifting all below it up one
        for idx in range(j, NUM_PERMUTATIONS - (i+1)):
            permutation_set[idx] = permutation_set[idx+1]
        # Replace last element with 0
        permutation_set[NUM_PERMUTATIONS - (i+1)] = np.zeros((1, num_moving), dtype=np.uint8)
        # Calculate hamming distance
        a1 = time.time()
        for j in range(i+1):
            for k in range(NUM_PERMUTATIONS-i):
                # Iterate over all in hamming_distance and permutation set
                # Calculate hamming dist of all combinations
                hamming_dist[j, k] = hamming_distance(max_hamming_set[j], permutation_set[k])
        b1 = time.time()
        print("Took {} seconds to calculate hamming distances".format(b1-a1))
        # Return index where max value was found
        print("Sum", np.sum(hamming_dist))
        j = np.argmax(np.sum(hamming_dist, axis=0))
        print("J val:", j)

        # Remove NUM_PERMUTATIONS-i column
        hamming_dist[:, NUM_PERMUTATIONS - (i+1)] = np.zeros((N), dtype=np.uint8)
        b = time.time()
        print("Took {} seconds to do one loop".format(b-a))

    return max_hamming_set


# ------Hamming tests


# @jit
def hamming_distance(array1, array2):
    """Calculate hamming distance between 2 arrays
    i.e how different they are."""
    if (array1.shape != array2.shape):
        raise ValueError("Input arrays must have same shape!")
    distance = 0
    for i in range(array1.shape[0]):
        if (array1[i] != array2[i]):
            distance += 1
    return distance


def gen_max_hamming_set(N, moving_cells, hamming_file):
    # My hamming function
    num_moving = len(moving_cells)
    NUM_PERM = 10000
    # This is the set of 10000 perms generated through Tom's method
    permutation_set = pd.read_csv(hamming_file, sep=",", header=None)
    # Convert to np array so no need to change rest of code
    permutation_set = permutation_set.values
    """
    # Create set containing 1M permutations
    permutation_set = np.zeros((NUM_PERM, num_moving), dtype=np.uint8)
    for i in range(NUM_PERM):
        permutation_set[i] = larger_set.loc[i]
    """
    max_dist_set = np.zeros((N, num_moving), dtype=np.uint8)
    hamming_dist = np.zeros((N, NUM_PERM), dtype=np.uint8)
    idx = random.randint(0, NUM_PERM)
    for i in range(N):
        a = time.time()
        max_dist_set[i] = permutation_set[idx]
        a1 = time.time()
        # for j in range(i+1):
        for j in range(i+1):
            # test in range(i)
            for k in range(NUM_PERM):
                hamming_dist[j, k] = hamming_distance(max_dist_set[j], permutation_set[k])
        b1 = time.time()
        print("Took {} seconds to calculate hamming distances".format(b1-a1))
        idx = np.argmax(np.sum(hamming_dist, axis=0))
        b = time.time()
        print("Took {} seconds to do one loop".format(b-a))
    return max_dist_set, hamming_dist


def TOM_gen_max_hamming_set(N, moving_cells):
    permutation = []
    iterations = 1000000
    for i in range(N):
        permutation.append(moving_cells)

    moving_cells_array = np.asarray(moving_cells)
    permutation_array = np.asarray(permutation)
    hamming_dist = []
    for j in range(iterations):
        for k in range(N):
            current_permutation = np.random.permutation(moving_cells)
            current_permutation_array = np.asarray(current_permutation)

            if hamming_distance(current_permutation_array, moving_cells_array) > hamming_distance(permutation_array[k], moving_cells_array):
                permutation_array[k] = current_permutation
                hamming_dist.append(hamming_distance(current_permutation_array, moving_cells_array))

    return permutation_array, hamming_dist


def main(argv=None):
    N = 1000
    avail_keys = [n for n in range(23)]
    print(avail_keys)
    max_dist_set, dist_array = gen_max_hamming_set(
        N, avail_keys, "/hepgpu3-data1/heyst/MPhys/CNN/hamming_set_10000.txt")
    """
    np.savetxt("hamming_set_10000.txt", max_dist_set, delimiter=",", fmt='%1.2i')
    np.savetxt("hamming_min_mean_10000.txt", dist_array, delimiter=",", fmt='%1.2i')
    """
    np.savetxt("mixed_hamming_set.txt", max_dist_set, delimiter=",", fmt='%1.2i')
    np.savetxt("mixed_hamming_min_mean.txt", dist_array, delimiter=",", fmt='%1.2i')
    """

    hamming_file = './hamming_set.txt'
    df1 = pd.read_csv(hamming_file, sep=",", header=None)
    print(df1)
    df = df1.loc[0, :].values
    print("Sliced", df)

    """


if __name__ == '__main__':
    main()
