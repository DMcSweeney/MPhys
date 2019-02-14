"""Training CNN for registration in Keras. Assumes all inputs are same shape."""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D
from keras.layers import Conv3DTranspose, BatchNormalization
from keras.models import Model
from keras.initializers import RandomNormal
from keras.utils import plot_model
import dataLoader as load
import numpy as np

# If on server
"""
fixed_dir = "/hepgpu3-data1/dmcsween/DataSplit/TrainingSet/PCT"
moving_dir = "/hepgpu3-data1/dmcsween/DataSplit/TrainingSet/PET"
dvf_dir = "/hepgpu3-data1/dmcsween/DataSplit/TrainingSet/DVF"

# If on laptop
fixed_dir = "E:/MPhys/DataSplit/TrainingSet/PCT"
moving_dir = "E:/MPhys/DataSplit/TrainingSet/PET"
dvf_dir = "E:/MPhys/DataSplit/TrainingSet/DVF"
"""
#IF your name is Tom 
fixed_dir = "D:\\Mphys\\Nifty\\PCT"
moving_dir = "D:\\Mphys\\Nifty\\PET"
dvf_dir = "D:\\Mphys\\Nifty\\DVF"




def shuffle_inplace(fixed, moving, dvf):
    assert len(fixed[:, ...]) == len(moving[:, ...]) == len(dvf[:, ...])
    print(fixed.shape)
    p = np.random.permutation(len(fixed[:, ...]))
    return fixed[p], moving[p], dvf[p]


def generator(inputs, label, batch_size=4):
    x_dim, y_dim, z_dim = inputs[0].shape[1:4]
    batch_inputs = [np.zeros((batch_size, x_dim, y_dim, z_dim)),
                    np.zeros((batch_size, x_dim, y_dim, z_dim))]
    # add 3 due to 3D vector
    batch_label = np.zeros((batch_size, x_dim, y_dim, z_dim, 3))
    print("Len label:", len(label))
    print("label shape:", label.shape)
    while True:
        for i in range(batch_size):
            # Random index from dataset
            index = np.random.choice(len(label), 1)
            print(inputs[0].shape, inputs[1].shape)
            batch_inputs[i] = [inputs[0][index, ...], inputs[1][index, ...]]
            batch_label[i] = label[index]
        yield batch_inputs, batch_label


def train():
    # Load DATA
    fixed_image, moving_image, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)

    # Turn into numpy arrays
    fixed_array = fixed_image.get_data()
    moving_array = moving_image.get_data()
    dvf_array = dvf_label.get_data(is_image=False)
    print(fixed_array.dtype)
    # Shuffle arrays
    fixed_array, moving_array, dvf_array = shuffle_inplace(fixed_array, moving_array, dvf_array)

    # Split into validation and training set
    validation_fixed, validation_moving, validation_dvf, train_fixed, train_moving, train_dvf = load.split_data(
        fixed_array, moving_array, dvf_array, validation_ratio=0.2)

    print("PCT Shape:", train_fixed.shape)
    print("PET Shape:", train_moving.shape)
    print("DVF Shape:", train_dvf.shape)

    # CNN Structure
    fixed_image = Input(shape=(train_fixed.shape[1:]))  # Ignore batch but include channel
    moving_image = Input(shape=(train_moving.shape[1:]))
    input = concatenate([fixed_image, moving_image], axis=0)

    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='down_1a')(input)
    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='down_1b')(x1)
    x1 = BatchNormalization()(x1)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_1')(x1)

    x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='down_2a')(x)
    x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='down_2b')(x2)
    x2 = BatchNormalization()(x2)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_2')(x2)

    x3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='down_3a')(x)
    x3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='down_3b')(x3)
    x3 = BatchNormalization()(x3)

    x = UpSampling3D(size=(2, 2, 2), name='UpSamp_3')(x3)
    y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same', name='Up_2a')(x)
    y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same', name='Up_2b')(y2)
    y2 = BatchNormalization()(y2)

    merge2 = concatenate([x2, y2], axis=0)

    x = UpSampling3D(size=(2, 2, 2), name='UpSamp_2')(merge2)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same', name='Up_1a')(x)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same', name='Up_1b')(y1)
    y1 = BatchNormalization()(y1)

    merge1 = concatenate([x1, y1], axis=0)

    # flat = Flatten()(merge1)
    # dense1 = Dense(1, activation='relu')(flat)
    # dense2 = Dense(dvf_params, activation='softmax')(flat)

    # Transform into flow field (from VoxelMorph Github)
    dvf = Conv3D(3, kernel_size=3, padding='same', name='dvf',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(merge1)

    # Train
    model = Model(inputs=[fixed_image, moving_image], outputs=dvf)
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    print(model.summary())
    plot_model(model, to_file='model.png')

    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
    model.fit_generator(generator([train_fixed, train_moving], train_dvf,
                                  batch_size=4), samples_per_epoch=50, nb_epoch=20, verbose=1)
    model.save('model.h5')
    accuracy = model.evaluate(x=[validation_fixed, validation_moving],
                              y=validation_dvf, batch_size=4)
    print("Accuracy:", accuracy[1])


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
