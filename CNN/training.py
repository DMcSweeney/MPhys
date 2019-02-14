"""Training CNN for registration in Keras. Assumes all inputs are same shape."""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D
from keras.layers import Conv3DTranspose, BatchNormalization
from keras.models import Model
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
import dataLoader as load
import numpy as np
import math

# If on server
fixed_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PlanningCT"
moving_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PET_Rigid"
dvf_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/DVF"

# If on laptop
"""
fixed_dir = "E:/MPhys/DataSplit/TrainingSet/PCT"
moving_dir = "E:/MPhys/DataSplit/TrainingSet/PET"
dvf_dir = "E:/MPhys/DataSplit/TrainingSet/DVF"
"""

batch_size = 4


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def shuffle_inplace(fixed, moving, dvf):
    assert len(fixed[:, ...]) == len(moving[:, ...]) == len(dvf[:, ...])
    p = np.random.permutation(len(fixed[:, ...]))
    return fixed[p], moving[p], dvf[p]


def generator(inputs, label, batch_size=3):
    x_dim, y_dim, z_dim, channel = inputs[0].shape[1:]
    fixed_input, moving_input = inputs
    batch_fixed, batch_moving = np.zeros((batch_size, x_dim, y_dim, z_dim, channel)), np.zeros(
        (batch_size, x_dim, y_dim, z_dim, channel))
    # add 3 due to 3D vector
    batch_label = np.zeros((batch_size, x_dim, y_dim, z_dim, 3))
    while True:
        for i in range(batch_size):
            # Random index from dataset
            index = np.random.choice(len(label), 1)
            batch_fixed[i], batch_moving[i] = inputs[0][index, ...], inputs[1][index, ...]
            batch_label[i] = label[index, ...]
        yield ({'input_1': batch_fixed, 'input_2': batch_moving}, {'dvf': batch_label})


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

    input = concatenate([fixed_image, moving_image])

    x1 = Conv3D(16, (7, 7, 7), strides=2, activation='relu',
                padding='same', name='downsample')(input)
    x1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='down_1a')(x1)
    x1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='down_1b')(x1)
    x1 = BatchNormalization()(x1)

    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='down_1c')(x1)
    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='down_1d')(x1)
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

    merge2 = concatenate([x2, y2])

    x = UpSampling3D(size=(2, 2, 2), name='UpSamp_2')(merge2)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same', name='Up_1a')(x)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same', name='Up_1b')(y1)
    y1 = BatchNormalization()(y1)

    merge1 = concatenate([x1, y1])

    # Transform into flow field (from VoxelMorph Github)
    dvf = Conv3DTranspose(64, (3, 3, 3), strides=2, activation='relu', padding='same',
                          name='upsample_dvf')(merge1)
    dvf = Conv3D(64, kernel_size=3, padding='same', name='dvf_64features',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(dvf)
    dvf = Conv3D(3, kernel_size=1, padding='same', name='dvf',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(dvf)
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2,
                                  patience=5, min_lr=0.001)
    history = LossHistory()
    checkpoint = ModelCheckpoint('best_model.h5', monitor='acc',
                                 verbose=1, save_best_only=True, period=1)

    callbacks = [reduce_lr, history, checkpoint]
    # Train
    model = Model(inputs=[fixed_image, moving_image], outputs=dvf)
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    # print(model.summary())
    plot_model(model, to_file='model.png')

    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
    model.fit_generator(generator=generator(inputs=[train_fixed, train_moving], label=train_dvf, batch_size=batch_size),
                        steps_per_epoch=math.ceil(train_fixed.shape[0]/batch_size),
                        epochs=100, verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(
                            inputs=[validation_fixed, validation_moving], label=validation_dvf),
                        validation_steps=math.ceil(validation_fixed[0]/batch_size),
                        validation_freq=2,
                        use_multiprocessing=True)

    # accuracy = model.evaluate_generator(generator(
    #    inputs=[validation_fixed, validation_moving], label=validation_dvf, batch_size=batch_size), steps=1, verbose=1)
    model.save('model.h5')
    print("Accuracy:", accuracy[1])


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
