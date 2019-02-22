""" File with useful functions to call"""
import numpy as np
import nibabel as nib
import os
import time


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
                    if np.random.uniform() > 0.5:
                        batch_fixed = Helpers.flip(input=batch_fixed)
                        batch_moving = Helpers.flip(input=batch_moving)
                    batch_fixed = Helpers.noise(batch_fixed, batch_size)
                    batch_moving = Helpers.noise(batch_moving, batch_size)
                    batch_fixed = Helpers.normalise(batch_fixed)
                    batch_moving = Helpers.normalise(batch_moving)
                yield ({'input_1': batch_fixed, 'input_2': batch_moving}, {'dvf': batch_label})
        else:
            while True:
                for i in range(batch_size):
                    # Random index from dataset
                    index = np.random.choice(len(label), 1)
                    batch_fixed[i], batch_moving[i] = inputs[0][index, ...], inputs[1][index, ...]
                    batch_fixed = Helpers.normalise(batch_fixed)
                    batch_moving = Helpers.normalise(batch_moving)
                yield ({'input_1': batch_fixed, 'input_2': batch_moving})

    @staticmethod
    def normalise(input):
        maxi = max(input)
        mini = min(input)
        med_maxi = np.median(maxi)
        med_mini = np.median(mini)
        normal_input = (input-med_mini)/(med_maxi-med_mini)
        return np.clip(normal_input, a_min=0, a_max=1)

    @staticmethod
    def flip(input):
        return np.flip(input, axis=1)

    @staticmethod
    def noise(input, batch_size):
        random = np.random.uniform()
        var_original = np.var(input[:, 100:110, 10:20, 60:70])
        var_multiplied = np.var(random*input[:, 100:110, 10:20, 60:70])
        var_noise = var_multiplied - random**2 * var_original
        std_noise = np.sqrt(abs(var_noise))
        noise_field = np.random.normal(loc=0, scale=std_noise, size=input.shape)
        return input + noise_field

    @staticmethod
    def shuffle_inplace(fixed, moving, dvf):
        np.random.seed(1234)
        assert len(fixed[:, ...]) == len(moving[:, ...]) == len(dvf[:, ...])
        p = np.random.permutation(len(fixed[:, ...]))
        np.random.seed(int(time.time()))
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
            [nib.save(nib.Nifti1Image(input_[idx, ...], base_affine[idx, ...]),
                      os.path.join(file_path,
                                   file_prefix + '%s.nii' % idx))
             for idx in range(batch_size)]
