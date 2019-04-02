from keras import optimizers
from keras.models import load_model
import dataLoader as load
import dataGenerator as gen
import pandas as pd


def infer(batch_size=2):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
    print("Load Data")
    image_data, __image, __label = load.data_reader(image_dir, image_dir, image_dir)

    image_array, image_affine = image_data.get_data()
    moving_array, moving_affine = __image.get_data()
    dvf_array, dvf_affine = __label.get_data()

    list_avail_keys = help.get_moveable_keys(image_array)
    # Get hamming set
    print("Load hamming Set")
    hamming_set = pd.read_csv("hamming_set.txt", sep=",", header=None)
    print(hamming_set)
    # Ignore moving and dvf
    validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
        image_array, moving_array, dvf_array, split_ratio=0.15)

    print('Load models')
    idx_list = [0, 1]
    model = load_model('model.h5')
    opt = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    output = model.predict_generator(generator=gen.predict_generator(
        validation_dataset, list_avail_keys, hamming_set, hamming_idx=idx_list, batch_size=batch_size, N=10), steps=3, verbose=1)
    print(output)


def main(argv=None):
    infer()


if __name__ == '__main__':
    main()
