"""
Load data for training and validation


"""
import os
import nibabel as nib
import numpy as np


def data_reader(fixed_dir, moving_dir, dvf_dir):

    fixed_image = DataLoader(fixed_dir)
    moving_image = DataLoader(moving_dir)
    dvf_label = DataLoader(dvf_dir)

    return fixed_image, moving_image, dvf_label


class DataLoader:

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.files = os.listdir(dir_name)
        self.files.sort()
        self.num_data = len(self.files)

        self.file_objects = [nib.load(os.path.join(dir_name, self.files[i]))
                             for i in range(self.num_data)]

        self.getaffine = [self.file_objects[i].affine for i in range(self.num_data)]

        self.num_labels = [self.file_objects[i].shape[3] if len(self.file_objects[i].shape) == 4
                           else 1 for i in range(self.num_data)]

        self.data_shape = list(np.shape(self.file_objects[0].dataobj))
        self.flatten = self.get_data().reshape(self.num_data, -1)

    def get_data(self, is_image=True):
        # Get data in form of array
        if is_image is True:
            # Load images
            array = [np.array(self.file_objects[i].dataobj) for i in range(self.num_data)]
            out = np.expand_dims(np.stack(array, axis=0), axis=-1)
        else:
            # Load DVF (label)
            array = np.squeeze([np.array(self.file_objects[i].dataobj)
                                for i in range(self.num_data)])
            out = np.expand_dims(np.stack(array, axis=0), axis=-1)
        return out


def split_data(fixed_image, moving_image, dvf_label, validation_ratio):
    np.random.shuffle(fixed_image)
    np.random.shuffle(moving_image)
    np.random.shuffle(dvf_label)
    validation_size = int(validation_ratio*fixed_image.shape[0])

    validation_fixed = fixed_image[:validation_size, ...]
    validation_moving = moving_image[:validation_size, ...]
    validation_dvf = np.squeeze(dvf_label[:validation_size, ...])

    train_fixed = fixed_image[validation_size:, ...]
    train_moving = moving_image[validation_size:, ...]
    train_dvf = np.squeeze(dvf_label[validation_size:, ...])
    print("Validation shape:", validation_fixed.shape)
    print("Training shape:", train_fixed.shape)
    return validation_fixed, validation_moving, validation_dvf, train_fixed, train_moving, train_dvf
