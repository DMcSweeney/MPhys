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
    #inputPath = "./all_logs/both_logs100perms"
    inputPath = './mixed_hamming_logs'
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

    hamming_set = hamming_set.loc[:9]
    # Ignore moving and dvf
    test_dataset, validation_moving, validation_dvf, trainVal_dataset, train_moving, train_dvf = helper.split_data(
        image_array, moving_array, dvf_array, split_ratio=0.05)
    print("Valid Shape:", test_dataset.shape)
    normalised_dataset = helper.normalise(test_dataset)
    print('Load models')
    idx_list = [3, 3]
    scores = []
    blank_idx = []
    K.clear_session()
    model = load_model(inputPath + '/best_model.h5')
    opt = optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    for i in range(23):
        blank_idx.append(i)
        print("Pre Eval")
        myPredictGen = gen.evaluate_generator(
            normalised_dataset, list_avail_keys, hamming_set, hamming_idx=idx_list, batch_size=batch_size, blank_idx=blank_idx, out_crop=False, inner_crop=False, N=10)
        accuracy = model.evaluate_generator(generator=myPredictGen, steps=5, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))
        scores.append(accuracy[1] * 100)
    for n, acc in enumerate(scores):
        print(n, acc)
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
