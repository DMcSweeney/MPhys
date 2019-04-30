"""Training CNN for registration in Keras. Assumes all inputs are same shape."""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D
from keras.layers import Conv3DTranspose, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from keras import optimizers
import dataLoader as load
import layers as myLayer
from customTensorBoard import TrainValTensorBoard
import helpers as helper
import math


# If on server
"""
fixed_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"

# Parameters to tweak
batch_size = 4

momentum = 0.75
"""
fixed_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PlanningCT"
moving_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PET_Rigid"
dvf_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/DVF"


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def TransferNet(input_shape, weights_path):
    activation = 'relu'

    input_layer = Input(shape=input_shape)
    Conv1 = Conv3D(32, (5, 5, 5), activation=activation, padding='same', name='Conv1')(input_layer)
    #x = BatchNormalization()(x)
    x = MaxPooling3D()(Conv1)
    Conv2 = Conv3D(64, (3, 3, 3), activation=activation, padding='same', name='Conv2')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling3D()(Conv2)
    Conv3 = Conv3D(128, (3, 3, 3), activation=activation, padding='same', name='Conv3')(x)
    downPath = Model(inputs=input_layer, outputs=Conv3)
    downPath.load_weights(weights_path, by_name=True)
    return downPath


def buildNet(input_shape, fixed_weights='./all_logs/PCT_logs100perms/final_model.h5', moving_weights='./all_logs/PET_logs100perms/final_model.h5'):
    activation = 'relu'
    fixed_img_input = Input(shape=input_shape)
    moving_img_input = Input(shape=input_shape)
    fixed_path = TransferNet(input_shape, fixed_weights)
    moving_path = TransferNet(input_shape, moving_weights)
    """
    mergeConv1 = concatenate([fixed_Conv1, moving_Conv1])
    mergeConv2 = concatenate([fixed_Conv2, moving_Conv2])
    mergeConv3 = concatenate([fixed_Conv3, moving_Conv3])
    """
    # Correlation layers
    # correlation_out = myLayer.correlation_layer(
    #    fixed_path(fixed_img_input), moving_path(moving_img_input), shape=input_shape[:-1], max_displacement=20, stride=2)
    correlation_out = concatenate([fixed_path(fixed_img_input), moving_path(moving_img_input)])
    x = Conv3DTranspose(128, (3, 3, 3), activation=activation,
                        padding='same', name='ConvUp3')(correlation_out)
    #merge3 = concatenate([x, mergeConv3])
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(64, (3, 3, 3), activation=activation, padding='same', name='ConvUp2')(x)
    #merge2 = concatenate([x, mergeConv2])
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(32, (5, 5, 5), activation=activation, padding='same', name='ConvUp1')(x)
    #merge1 = concatenate([x, mergeConv1])
    dvf = Conv3D(64, kernel_size=3, activation=activation,
                 padding='same', name='dvf_64features')(x)
    dvf = Conv3D(3, kernel_size=1, activation=None, padding='same', name='dvf')(dvf)
    model = Model(inputs=[fixed_img_input, moving_img_input], outputs=dvf)
    return model


def train(batch_size=2):
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
    outputPath = './transfer_logs/'
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5)
    history = LossHistory()
    checkpoint = ModelCheckpoint(outputPath + 'best_model.h5', monitor='val_loss',
                                 verbose=1, save_best_only=True, period=1)
    tensorboard = TrainValTensorBoard(write_graph=False, log_dir=outputPath)
    callbacks = [reduce_lr, history, checkpoint, tensorboard]

    # Train
    model = buildNet(train_fixed.shape[1:])
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    # print(model.summary())
    plot_model(model, to_file=outputPath + 'model.png', show_shapes=True)
    opt = optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt, loss='mean_squared_error')
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


def infer():
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
