"""Inference file for CNN registration on unseen data"""
from keras.models import load_model
from helpers import Helpers
import dataLoader as load
import math

helper = Helpers()


fixed_dir = "E:/MPhys/DataSplit/TrainingSet/PCT"
moving_dir = "E:/MPhys/DataSplit/TrainingSet/PET"
dvf_dir = "E:/MPhys/DataSplit/TrainingSet/DVF"

batch_size = 2


def inference():
    # Load data to Transform
    fixed_predict, moving_predict, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)

    # Turn into numpy arrays
    fixed_array = fixed_predict.get_data()
    moving_array = moving_predict.get_data()
    dvf_array = dvf_label.get_data(is_image=False)

    # Shuffle
    fixed_array, moving_array, dvf_array = helper.shuffle_inplace(
        fixed_array, moving_array, dvf_array)
    # Split into test/training data
    test_fixed, test_moving, test_dvf, train_fixed, train_moving, train_dvf = helper.split_data(
        fixed_array, moving_array, dvf_array, split_ratio=0.05)

    # Load models
    model = load_model('best_model.h5')
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
    dvf = model.predict_generator(helper.generator(
        inputs=[test_fixed, test_moving], label=test_dvf, predict=True),
        steps=math.ceil(test_fixed.shape[0]/batch_size),
        verbose=1)
    # Save DVF
    helper.write_images(dvf, test_fixed, file_path='./outputs/', file_prefix='dvf')
    # Warp image

    # Save warped


def main(argv=None):
    inference()


if __name__ == '__main__':
    main()
