from keras.layers import (Dense, Dropout, Concatenate, Input, Activation, Flatten, Conv3D,
                          MaxPooling3d, GlobalAveragePooling3D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import optimizers
from time import strftime, localtime
import warnings
import os
import pickle
import resnetBottom
from DataGenerator import DataGenerator
import numpy as np
import h5py
#  from keras.utils import plot_model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def trian(tileSize=64, numPuzzles=9):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
    moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"
    dvf_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/DVF"

    image_data, __image, __label = load.data_reader(image_dir, moving_dir, dvf_dir)

    image_array, image_affine = image_data_image.get_data()

    fixed_image = Input(shape=(train_fixed.shape[1:]))


    validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
        fixed_image, __image, __label, split_ratio=0.15)

    image = Input(shape=(train_dataset.shape[1:]))  

    x = Conv3D(64, (7, 7, 7), strides=2, padding='same')(inputTensor)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=2, padding='same')(x)

    x = Conv3D(64, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv3D(128, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv3D(256, (3, 3 ,3), strides=2 , padding='same')(x)
    x = Activation('relu')(x)

    x = Conv3D(512, (3, 3, 3), strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling3D()(x)

    model = Model(image, x, name='Jigsaw_Model')

    dataGenerator = DataGenerator(batchSize=batch_size, meanTensor=normalize_mean,
                              stdTensor=normalize_std, maxHammingSet=max_hamming_set[:hamming_set_size])

    # Output all data from a training session into a dated folder
    outputPath = './model_data/{}'.format(strftime('%b_%d_%H:%M:%S', localtime()))
    os.makedirs(outputPath)
    checkpointer = ModelCheckpoint(
        outputPath +
        '/weights_improvement.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True)
    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss', patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    # tBoardLogger = TensorBoard(log_dir=outputPath, histogram_freq=5,
    # batch_size=batch_size, write_graph=True, write_grads=True,
    # write_images=True)

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=dataGenerator.generate(train_dataset),
                                  epochs=num_epochs,
                                  steps_per_epoch=train_dataset.shape[0] // batch_size,
                                  validation_data=dataGenerator.generate(
                                      val_dataset),
                                  validation_steps=val_dataset.shape[0] // batch_size,
                                  use_multiprocessing=USE_MULTIPROCESSING,
                                  workers=n_workers,
                                  callbacks=[checkpointer, reduce_lr_plateau, early_stop])

    scores = model.evaluate_generator(
        dataGenerator.generate(test_dataset),
        steps=test_dataset.shape[0] //
        batch_size,
        workers=n_workers,
        use_multiprocessing=USE_MULTIPROCESSING)



def main(argv=None):
    train()


if __name__ == '__main__':
    main()
