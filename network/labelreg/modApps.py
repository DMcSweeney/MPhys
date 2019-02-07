"""
Modified script to tet warp volumes function
"""
import tensorflow as tf
import utils as util
import helpers as helper

moving_path = 'E:\\MPhys\\DataSplit\\ValidationSet\\PET'
ddf_path = 'E:\\MPhys\\DataSplit\\ValidationSet\\DVF'
save_path = 'E:\\MPhys'


def warp_volumes_by_ddf(input_, ddf):
    print(ddf.shape[1:4])
    grid_warped = util.get_reference_grid(ddf.shape[1:4]) + ddf
    warped = util.resample_linear(tf.convert_to_tensor(input_, dtype=tf.float32), grid_warped)
    with tf.Session() as sess:
        return sess.run(warped)


def main(argv=None):
    moving_image, ddf, _ = helper.get_data_readers(moving_path, ddf_path)
    # Expand dims for batches
    print(moving_image.get_data().shape)
    warped_image = warp_volumes_by_ddf(moving_image.get_data(), ddf.get_data())
    helper.write_images(warped_image, save_path, 'warped_image')


if __name__ == '__main__':
    main()
