"""Inference file for CNN registration on unseen data"""
from keras.models import load_model
import helpers as helper
import dataLoader as load
import math


# On server
fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"
"""
# On laptop
fixed_dir = "E:/MPhys/Data128/PlanningCT"
moving_dir = "E:/MPhys/Data128/PET_Rigid"
dvf_dir = "E:/MPhys/Data128/DVF"
"""
batch_size = 1


def inference():
    print('Load data to Transform')
    fixed_predict, moving_predict, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)

    print('Turn into numpy arrays')
    fixed_array, fixed_affine = fixed_predict.get_data()
    moving_array, moving_affine = moving_predict.get_data()
    dvf_array, dvf_affine = dvf_label.get_data(is_image=False)

    print('Shuffle')
    fixed_array, moving_array, dvf_array = helper.shuffle_inplace(
        fixed_array, moving_array, dvf_array)
    fixed_affine, moving_affine, dvf_affine = helper.shuffle_inplace(
        fixed_affine, moving_affine, dvf_affine)

    print('Split into test/training data')
    test_fixed, test_moving, test_dvf, train_fixed, train_moving, train_dvf = helper.split_data(
        fixed_array, moving_array, dvf_array, split_ratio=0.05)
    test_fixed_affine, test_moving_affine, test_dvf_affine, train_fixed_affine, train_moving_affine, train_dvf_affine = helper.split_data(
        fixed_affine, moving_affine, dvf_affine, split_ratio=0.05)

    print('Load models')
    print("Fixed input", test_fixed.shape)
    print("Moving input", test_moving.shape)
    model = load_model('best_model.h5')
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
    dvf = model.predict_generator(helper.generator([test_fixed, test_moving], label=test_dvf, predict=True, batch_size=batch_size), steps=math.ceil(
        test_fixed.shape[0]/batch_size), verbose=1)
    test_loss = wmodel.evaluate_generator(helper.generator([test_fixed, test_moving], label=test_dvf, predict=True, batch_size=batch_size), steps=math.ceil(
        test_fixed.shape[0]/batch_size), verbose=1)

    print('Save DVF')
    # Save images
    helper.write_images(test_fixed, test_fixed_affine, file_path='./outputs/', file_prefix='fixed')
    helper.write_images(test_moving, test_moving_affine,
                        file_path='./outputs/', file_prefix='moving')
    helper.write_images(dvf, test_fixed_affine, file_path='./outputs/', file_prefix='dvf')
    print("Test Loss:", test_loss)
    # Save warped
    print("Test Loss Shape:", test_loss.shape)


def main(argv=None):
    inference()


if __name__ == '__main__':
    main()
