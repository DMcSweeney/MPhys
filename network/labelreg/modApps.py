"""
Modified script to tet warp volumes function
"""
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import utils as util
import helpers as helper

moving_path = 'E:\\MPhys\\DataSplit\\ValidationSet\\PET\\HN-CHUM-001'
ddf_path = 'E:\\MPhys\\DataSplit\\ValidationSet\\DVF\\HN-CHUM-001'
save_path = 'E:\\MPhys'


def warp_volumes_by_ddf(input_, ddf):
    print(ddf.shape[1:4])
    grid_warped = util.get_reference_grid(ddf.shape[1:4]) + ddf
    warped = util.resample_linear(tf.convert_to_tensor(input_, dtype=tf.float32), grid_warped)
    with tf.Session() as sess:
        return sess.run(warped)


def main(argv=None):
    moving_array = sitk.GetArrayFromImage(sitk.ReadImage(moving_path))
    ddf_array = sitk.GetArrayFromImage(sitk.ReadImage(ddf_path))
    # Expand dims for batches
    moving_array = np.expand_dims(moving_array, axis=0)
    ddf_array = np.expand_dims(ddf_array, axis=0)
    warped_image = warp_volumes_by_ddf(moving_array, ddf_array)
    helper.write_images(warped_images, save_path, 'warped_image')


if __name__ == '__main__':
    main()
