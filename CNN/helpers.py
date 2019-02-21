""" File with useful functions to call"""
import numpy as np
import nibabel as nib
import os


class Helpers(object):
    @staticmethod
    def generator(inputs, label, batch_size=4, predict=False):
        x_dim, y_dim, z_dim, channel = inputs[0].shape[1:]
        fixed_input, moving_input = inputs
        batch_fixed, batch_moving = np.zeros((batch_size, x_dim, y_dim, z_dim, channel)), np.zeros(
            (batch_size, x_dim, y_dim, z_dim, channel))
        # add 3 due to 3D vector
        batch_label = np.zeros((batch_size, x_dim, y_dim, z_dim, 3))
        if predict is False:
            while True:
                for i in range(batch_size):
                    # Random index from dataset
                    index = np.random.choice(len(label), 1)
                    batch_fixed[i], batch_moving[i] = inputs[0][index, ...], inputs[1][index, ...]
                    batch_label[i] = label[index, ...]
                yield ({'input_1': batch_fixed, 'input_2': batch_moving}, {'dvf': batch_label})
        else:
            while True:
                for i in range(batch_size):
                    # Random index from dataset
                    index = np.random.choice(len(label), 1)
                    batch_fixed[i], batch_moving[i] = inputs[0][index, ...], inputs[1][index, ...]
                yield ({'input_1': batch_fixed, 'input_2': batch_moving})

    @staticmethod
    def shuffle_inplace(fixed, moving, dvf):
        np.random.seed(1234)
        assert len(fixed[:, ...]) == len(moving[:, ...]) == len(dvf[:, ...])
        p = np.random.permutation(len(fixed[:, ...]))
        return fixed[p], moving[p], dvf[p]

    @staticmethod
    def split_data(fixed_image, moving_image, dvf_label, split_ratio):
        validation_size = int(split_ratio*fixed_image.shape[0])

        validation_fixed = fixed_image[:validation_size, ...]
        validation_moving = moving_image[:validation_size, ...]
        validation_dvf = np.squeeze(dvf_label[:validation_size, ...])

        train_fixed = fixed_image[validation_size:, ...]
        train_moving = moving_image[validation_size:, ...]
        train_dvf = np.squeeze(dvf_label[validation_size:, ...])
        print("Validation shape:", validation_fixed.shape)
        print("Training shape:", train_fixed.shape)
        return validation_fixed, validation_moving, validation_dvf, train_fixed, train_moving, train_dvf

    @staticmethod
    def write_images(input_, base_affine, file_path=None, file_prefix=''):
        if file_path is not None:
            batch_size = input_.shape[0]
            print("affine shape", base_affine.shape)
            print("Input shape", input_.shape)
            #affine = [base_affine[idx] for idx in range(batch_size)]
            [nib.save(nib.Nifti1Image(input_[idx, ...], base_affine[idx, ...]),
                      os.path.join(file_path,
                                   file_prefix + '%s.nii' % idx))
             for idx in range(batch_size)]
