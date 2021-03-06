"""Training CNN for registration in Keras. Assumes all inputs are same shape."""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D
from keras.layers import Conv3DTranspose, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
import dataLoader as load
import layers as myLayer
from customTensorBoard import TrainValTensorBoard
import helpers as helper
import math


# If on server
fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"
"""
# If on laptop
fixed_dir = "E:/MPhys/DataSplit/TrainingSet/PCT"
moving_dir = "E:/MPhys/DataSpit/TrainingSet/PET"
dvf_dir = "E:/MPhys/DataSplit/TrainingSet/DVF"
"""
# Parameters to tweak
batch_size = 4
activation = 'relu'
momentum = 0.75


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train():
    # Load DATA
    fixed_image, moving_image, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)

    # Turn into numpy arrays
    fixed_array, fixed_affine = fixed_image.get_data()
    moving_array, moving_affine = moving_image.get_data()
    dvf_array, dvf_affine = dvf_label.get_data(is_image=False)
    # Shuffle arrays
    fixed_array, moving_array, dvf_array = helper.shuffle_inplace(
        fixed_array, moving_array, dvf_array)
    fixed_affine, moving_affine, dvf_affine = helper.shuffle_inplace(
        fixed_affine, moving_affine, dvf_affine)
    # Split into test and training set
    # Training/Validation/Test = 80/15/5 split
    test_fixed, test_moving, test_dvf, train_fixed, train_moving, train_dvf = helper.split_data(
        fixed_array, moving_array, dvf_array, split_ratio=0.05)
    # Test affine
    test_fixed_affine, test_moving_affine, test_dvf_affine, train_fixed_affine, train_moving_affine, train_dvf_affine = helper.split_data(
        fixed_affine, moving_affine, dvf_affine, split_ratio=0.05)
    # Split training into validation and training set
    validation_fixed, validation_moving, validation_dvf, train_fixed, train_moving, train_dvf = helper.split_data(
        train_fixed, train_moving, train_dvf, split_ratio=0.15)

    print("PCT Shape:", train_fixed.shape)
    print("PET Shape:", train_moving.shape)
    print("DVF Shape:", train_dvf.shape)

    # CNN Structure
    fixed_image = Input(shape=(train_fixed.shape[1:]))  # Ignore batch but include channel
    moving_image = Input(shape=(train_moving.shape[1:]))

    # Correlation layers
    correlation_out = myLayer.correlation_layer(
        fixed_image, moving_image, shape=train_fixed.shape[1:4], max_displacement=20, stride=2)

    x1 = Conv3D(64, (3, 3, 3), strides=2, activation=activation,
                padding='same', name='downsample1')(correlation_out)
    x1 = Conv3D(32, (3, 3, 3), strides=2, activation=activation,
                padding='same', name='downsample2')(x1)
    x1 = Conv3D(16, (3, 3, 3), strides=2, activation=activation,
                padding='same', name='downsample3')(x1)
    x1 = BatchNormalization(axis=-1, momentum=momentum)(x1)

    x1 = Conv3D(64, (3, 3, 3), activation=activation, padding='same', name='down_1a')(x1)
    x1 = Conv3D(64, (3, 3, 3), activation=activation, padding='same', name='down_1b')(x1)
    x1 = Conv3D(64, (3, 3, 3), activation=activation, padding='same', name='down_1c')(x1)
    x1 = BatchNormalization(axis=-1, momentum=momentum)(x1)

    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_1')(x1)

    x2 = Conv3D(128, (3, 3, 3), activation=activation, padding='same', name='down_2a')(x)
    x2 = Conv3D(128, (3, 3, 3), activation=activation, padding='same', name='down_2b')(x2)
    x2 = Conv3D(128, (3, 3, 3), activation=activation, padding='same', name='down_2c')(x2)
    x2 = BatchNormalization(axis=-1, momentum=momentum)(x2)

    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_2')(x2)

    x3 = Conv3D(256, (3, 3, 3), activation=activation, padding='same', name='down_3a')(x)
    x3 = Conv3D(256, (3, 3, 3), activation=activation, padding='same', name='down_3b')(x3)
    x3 = BatchNormalization(axis=-1, momentum=momentum)(x3)

    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_3')(x3)

    x4 = Conv3D(512, (3, 3, 3), activation=activation, padding='same', name='down_4a')(x)

    x = UpSampling3D(size=(2, 2, 2), name='UpSamp_4')(x4)
    y3 = Conv3DTranspose(256, (3, 3, 3), activation=activation, padding='same', name='Up_3a')(x)
    y3 = Conv3DTranspose(256, (3, 3, 3), activation=activation, padding='same', name='Up_3b')(y3)
    y3 = Conv3DTranspose(256, (3, 3, 3), activation=activation, padding='same', name='Up_3c')(y3)
    y3 = BatchNormalization()(y3)

    merge3 = concatenate([x3, y3])

    x = UpSampling3D(size=(2, 2, 2), name='UpSamp_3')(merge3)
    y2 = Conv3DTranspose(128, (3, 3, 3), activation=activation, padding='same', name='Up_2a')(x)
    y2 = Conv3DTranspose(128, (3, 3, 3), activation=activation, padding='same', name='Up_2b')(y2)
    y2 = Conv3DTranspose(128, (3, 3, 3), activation=activation, padding='same', name='Up_2c')(y2)
    y2 = BatchNormalization(axis=-1, momentum=momentum)(y2)

    merge2 = concatenate([x2, y2])

    x = UpSampling3D(size=(2, 2, 2), name='UpSamp_2')(merge2)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation=activation, padding='same', name='Up_1a')(x)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation=activation, padding='same', name='Up_1b')(y1)
    y1 = Conv3DTranspose(64, (3, 3, 3), activation=activation, padding='same', name='Up_1c')(y1)
    y1 = BatchNormalization(axis=-1, momentum=momentum)(y1)

    merge1 = concatenate([x1, y1])

    # Transform into flow field (from VoxelMorph Github)
    upsample = Conv3DTranspose(64, (3, 3, 3), strides=2, activation=activation, padding='same',
                               name='upsample_dvf1')(merge1)
    upsample = Conv3DTranspose(64, (3, 3, 3), strides=2, activation=activation, padding='same',
                               name='upsample_dvf2')(upsample)
    upsample = Conv3DTranspose(64, (3, 3, 3), strides=2, activation=activation, padding='same',
                               name='upsample_dvf3')(upsample)
    upsample = BatchNormalization(axis=-1, momentum=momentum)(upsample)

    dvf = Conv3D(64, kernel_size=3, activation=activation,
                 padding='same', name='dvf_64features')(upsample)
    #dvf = Conv3D(3, kernel_size=3, activation=activation, padding='same', name='dvf')(dvf)
    dvf = Conv3D(3, kernel_size=1, activation=None, padding='same', name='dvf')(dvf)

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.00001)
    history = LossHistory()
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss',
                                 verbose=1, save_best_only=True, period=1)
    tensorboard = TrainValTensorBoard(write_graph=False)
    callbacks = [reduce_lr, history, checkpoint, tensorboard]

    # Train
    model = Model(inputs=[fixed_image, moving_image], outputs=dvf)
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    # print(model.summary())
    plot_model(model, to_file='model.png')
    #Adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit_generator(generator=helper.generator(inputs=[train_fixed, train_moving], label=train_dvf, batch_size=batch_size),
                        steps_per_epoch=math.ceil(train_fixed.shape[0]/batch_size),
                        epochs=75, verbose=1,
                        callbacks=callbacks,
                        validation_data=helper.generator(
                            inputs=[validation_fixed, validation_moving], label=validation_dvf, batch_size=batch_size),
                        validation_steps=math.ceil(validation_fixed.shape[0]/batch_size))

    # accuracy = model.evaluate_generator(generator(
    #    inputs=[validation_fixed, validation_moving], label=validation_dvf, batch_size=batch_size), steps=1, verbose=1)
    model.save('model.h5')

    """Testing to see where issue with DVF is """
    dvf = model.predict(helper.generator([test_fixed, test_moving], label=test_dvf, predict=True, batch_size=1), steps=math.ceil(
        test_fixed.shape[0]/batch_size), verbose=1)
    helper.write_images(test_fixed, test_fixed_affine, file_path='./outputs/', file_prefix='fixed')
    helper.write_images(test_moving, test_moving_affine,
                        file_path='./outputs/', file_prefix='moving')
    helper.write_images(dvf, test_fixed_affine, file_path='./outputs/', file_prefix='dvf')


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
