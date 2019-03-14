from keras.layers import (Dense, Dropout, Concatenate, Input, Activation, Flatten,
                          Conv3D, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras import optimizers
from time import strftime, localtime
import dataLoader as load
import os
import time
import hamming
import helpers as helper
from customTensorBoard import TrainValTensorBoard
import dataGenerator as gen
import JigsawHelpers as help

#  from keras.utils import plot_model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def basicModel(tileSize=32, numPuzzles=23):

    # CNN structure
    inputTensor = Input((tileSize, tileSize, tileSize, 1))

    x = Conv3D(64, (7, 7, 7), strides=2, padding='same')(inputTensor)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=2, padding='same')(x)

    x = Conv3D(64, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv3D(128, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv3D(256, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv3D(512, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    outputs = GlobalAveragePooling3D()(x)

    model = Model(inputs=inputTensor, outputs=outputs, name='Jigsaw_Model')

    return model


def trivialNet(numPuzzles=23, tileSize=32, hammingSetSize=25):

    inputShape = (tileSize, tileSize, tileSize, 1)
    modelInputs = [Input(inputShape) for _ in range(numPuzzles)]
    sharedLayer = basicModel()
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]
    x = Concatenate()(sharedLayers)  # Reconsider what axis to merge
    x = Dense(512, activation='relu')(x)
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


def train(tileSize=64, numPuzzles=23, num_permutations=25, batch_size=1):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"

    image_data, __image, __label = load.data_reader(image_dir, image_dir, image_dir)

    image_array, image_affine = image_data.get_data()
    moving_array, moving_affine = __image.get_data()
    dvf_array, dvf_affine = __label.get_data()

    list_avail_keys = help.get_moveable_keys(image_array)
    # Get hamming set
    start_time = time.time()
    hamming_set = hamming.gen_max_hamming_set(num_permutations, list_avail_keys)
    end_time = time.time()
    print("Took {} to generate {} permutations". format(
        end_time - start_time, num_permutations))

    # Ignore moving and dvf
    validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
        image_array, moving_array, dvf_array, split_ratio=0.15)

    # Output all data from a training session into a dated folder
    outputPath = "./logdir"
    # callbacks
    checkpointer = ModelCheckpoint(outputPath + '/weights_improvement.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)
    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tensorboard = TrainValTensorBoard(write_graph=False)
    callbacks = [checkpointer, reduce_lr_plateau, early_stop, tensorboard]
    # BUILD Model
    model = trivialNet()

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    print(model.summary())
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(generator=gen.generator(train_dataset, list_avail_keys, hamming_set),
                        epochs=100, verbose=1,
                        steps_per_epoch=train_dataset.shape[0] // batch_size,
                        validation_data=gen.generator(
        validation_dataset, list_avail_keys, hamming_set),
        validation_steps=validation_dataset.shape[0] // batch_size, callbacks=callbacks)


"""
    scores = model.evaluate_generator(
        dataGenerator.generate(test_dataset),
        steps=test_dataset.shape[0] //batch_size)
"""


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
