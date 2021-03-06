from keras import optimizers
from keras.models import load_model
from keras import backend as K
import dataLoader as load
import dataGenerator as gen
import JigsawHelpers as help
import helpers as helper
import pandas as pd
import numpy as np


def infer(batch_size=2):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
    #image_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PlanningCT"
    inputPath = "./all_logs/both_logs100perms"
    #inputPath = './mixed_hamming_logs'
    print("Load Data")
    image_data, __image, __label = load.data_reader(image_dir, image_dir, image_dir)

    image_array, image_affine = image_data.get_data()
    moving_array, moving_affine = __image.get_data()
    dvf_array, dvf_affine = __label.get_data()
    """
    list_avail_keys = help.get_moveable_keys(image_array)
    # Get hamming set
    print("Load hamming Set")
    hamming_set = pd.read_csv("hamming_set.txt", sep=",", header=None)
    print(hamming_set)
    """
    avail_keys = pd.read_csv("avail_keys_both.txt", sep=",", header=None)
    list_avail_keys = [(avail_keys.loc[i, 0], avail_keys.loc[i, 1], avail_keys.loc[i, 2])
                       for i in range(len(avail_keys))]
    # Get hamming set
    print("Load hamming Set")
    hamming_set = pd.read_csv(
        "mixed_hamming_set.txt", sep=",", header=None)

    hamming_set = hamming_set.loc[:99]
    # Ignore moving and dvf
    test_dataset, validation_moving, validation_dvf, trainVal_dataset, train_moving, train_dvf = helper.split_data(
        image_array, moving_array, dvf_array, split_ratio=0.05)
    print("Valid Shape:", test_dataset.shape)
    normalised_dataset = helper.normalise(test_dataset)
    print('Load models')
    scores = np.zeros((15, 20))
    blank_idx = [n for n in range(23)]
    print(blank_idx)
    K.clear_session()
    model = load_model(inputPath + '/final_model.h5')
    opt = optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    idx_list = []
    # i is border size
    for i in range(15):
        for j in range(20):
            idx_list = [10, 10]
            print("Pre Eval:", i, j)
            myPredictGen = gen.evaluate_generator(
                normalised_dataset, list_avail_keys, hamming_set, hamming_idx=idx_list, batch_size=batch_size, blank_idx=blank_idx, border_size=i, image_idx=[10, 10], full_crop=False, out_crop=True, inner_crop=False, N=100)
            accuracy = model.evaluate_generator(generator=myPredictGen, steps=5, verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))
            scores[i, j] = (accuracy[1] * 100)

    np.savetxt("scores_diff_border.txt", scores, delimiter=",", fmt='%1.2i')
    avg_score = np.mean(scores, axis=1)
    avg_perm = np.mean(scores, axis=0)
    error_score = np.std(scores, axis=1)
    error_perm = np.std(scores, axis=0)
    var_score = np.var(scores, axis=1)
    var_perm = np.var(scores, axis=0)
    print("Scores:", avg_score, error_score, var_score)
    print("Perms:", avg_perm, error_perm, var_perm)
    print("Done")


"""
output = model.predict_generator(generator=myPredictGen, steps=1, verbose=1)
for i, img in enumerate(output):
    print(img)
    print("Predicted index:{}. Should be: {}".format(np.argmax(img), idx_list[i]))
    print("Accuracy: {}".format(np.max(img)))
"""


def main(argv=None):
    infer()


if __name__ == '__main__':
    main()
