from keras.layers import (Dense, Concatenate, Input, Flatten,
                          Conv3D, MaxPooling3D, BatchNormalization)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model
import helpers as helper
import dataLoader as load
from customTensorBoard import TrainValTensorBoard
import dataGenerator as gen
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import (train_test_split, KFold)
from matplotlib import pyplot as plt
import os, sys
import numpy as np

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def createSharedAlexnet3D(tileSize=25, numPuzzles=23, hammingSetSize=10):

    input_layers = [Input(shape=(tileSize, tileSize, tileSize, 1),
                          name="alexnet_input_{}".format(n)) for n in range(numPuzzles)]
    conv1 = Conv3D(96, (11, 11, 11), strides=(4, 4, 4), activation='relu',
                   padding='valid', name="Convolutional_1")
    bn1 = BatchNormalization(name="BatchNorm_1")

    conv2 = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), activation='relu',
                   padding='same', name="Convolutional_2")
    bn2 = BatchNormalization(name="BatchNorm_2")

    maxPool1 = MaxPooling3D(strides=(2, 2, 2), padding='same', name="MaxPool_1")

    conv3 = Conv3D(384, (3, 3, 3), activation='relu', padding='same', name="Convolutional_3")
    bn3 = BatchNormalization(name="BatchNorm_3")

    maxPool2 = MaxPooling3D(strides=(2, 2, 2), padding='same', name="MaxPool_2")

    conv4 = Conv3D(384, (3, 3, 3), strides=(1, 1, 1), padding='same', name="Convolutional_4")
    bn4 = BatchNormalization(name="BatchNorm_4")

    conv5 = Conv3D(256, (3, 3, 3), padding='same', name="Convolutional_5")
    bn5 = BatchNormalization(name="BatchNorm_5")

    fc6 = Dense(4096, activation='relu', name="FullyConnected_1")
    fc7 = Dense(4096, activation='relu', name="FullyConnected_2")
    fc8 = Dense(hammingSetSize, activation='softmax', name="ClassificationOutput")

    cnvd1 = [conv1(a) for a in input_layers]
    bnd1 = [bn1(a) for a in cnvd1]

    cnvd2 = [conv2(a) for a in bnd1]
    bnd2 = [bn2(a) for a in cnvd2]

    mpd1 = [maxPool1(a) for a in bnd2]

    cnvd3 = [conv3(x) for x in mpd1]
    bnd3 = [bn3(x) for x in cnvd3]

    mpd2 = [maxPool2(x) for x in bnd3]

    cnvd4 = [conv4(x) for x in mpd2]
    bnd4 = [bn4(x) for x in cnvd4]

    cnvd5 = [conv5(x) for x in bnd4]
    bnd5 = [bn5(x) for x in cnvd5]

    fcd6 = [fc6(Flatten()(x)) for x in bnd5]

    concatd = Concatenate()(fcd6)

    fc7d = fc7(concatd)
    fc8d = fc8(fc7d)

    model = Model(inputs=input_layers, output=fc8d)

    return model


def createAlexnet3D(input_shape=(25, 25, 25, 1)):
    kernel_initializer = 'RandomNormal'
    activation = 'sigmoid'
    inputLayer = Input(shape=(input_shape))
    x = Conv3D(96, (11, 11, 11), strides=(2, 2, 2), activation=activation,
               padding='valid', kernel_initializer=kernel_initializer)(inputLayer)  # Note, strides are different!
    #x = BatchNormalization()(x)
    x = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), activation=activation,
               padding='same', kernel_initializer=kernel_initializer)(x)
    #x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)
    x = Conv3D(384, (3, 3, 3), activation=activation, padding='same',
               kernel_initializer=kernel_initializer)(x)
    #x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)
    x = Conv3D(384, (3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer=kernel_initializer, activation=activation)(x)
    #x = BatchNormalization()(x)
    x = Conv3D(256, (3, 3, 3), padding='same',
               kernel_initializer=kernel_initializer, activation=activation)(x)
    #x = BatchNormalization()(x)
    x = Flatten()(x)
    outputLayer = Dense(1024, activation=activation)(x)
    an3D = Model(inputs=[inputLayer], outputs=outputLayer)
    return an3D


def createNet(input_shape=(28, 28, 28, 1)):
    activation = 'relu'
    inputLayer = Input(shape=(input_shape))
    x = Conv3D(32, (5, 5, 5), activation=activation, padding='same', name='Conv1')(inputLayer)
    #x = BatchNormalization()(x)
    x = MaxPooling3D()(x)
    x = Conv3D(64, (3, 3, 3), activation=activation, padding='same', name='Conv2')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling3D()(x)
    x = Conv3D(128, (3, 3, 3), activation=activation, padding='same', name='Conv3')(x)
    #x = BatchNormalization()(x)
    x = Flatten()(x)
    outputLayer = Dense(1024, activation=activation)(x)
    an3D = Model(inputs=[inputLayer], outputs=outputLayer)
    return an3D


def createSharedAlexnet3D_onemodel(input_shape=(28, 28, 28, 1), nInputs=23, nclass=100):
    activation = 'relu'
    input_layers = [Input(shape=input_shape, name="alexnet_input_{}".format(n))
                    for n in range(nInputs)]
    #an3D = createAlexnet3D(input_shape)
    an3D = createNet(input_shape)
    fc6 = Concatenate()([an3D(x) for x in input_layers])
    fc7 = Dense(8192, activation=activation)(fc6)
    fc8 = Dense(nclass, activation='softmax', name="ClassificationOutput")(fc7)
    model = Model(inputs=input_layers, output=fc8)
    return model


def train(tileSize=64, numPuzzles=23, num_permutations=100, batch_size=16):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PlanningCT"
    #image_dir = "/hepgpu3-data1/dmcsween/Data128/ResampleData/PET_Rigid"

    print("Load Data")
    image_data, __image, __label = load.data_reader(image_dir, image_dir, image_dir)

    image_array, image_affine = image_data.get_data()
    moving_array, moving_affine = __image.get_data()
    dvf_array, dvf_affine = __label.get_data()
    """
    list_avail_keys = help.get_moveable_keys(image_array)
    hamming_set = pd.read_csv(
        "hamming_set_PCT.txt", sep=",", header=None)
    """
    avail_keys = pd.read_csv("avail_keys_both.txt", sep=",", header=None)
    print("Len keys:", len(avail_keys))
    list_avail_keys = [(avail_keys.loc[i, 0], avail_keys.loc[i, 1], avail_keys.loc[i, 2])
                       for i in range(len(avail_keys))]
    print(list_avail_keys)
    # Get hamming set
    print("Load hamming Set")
    hamming_set = pd.read_csv(
        "mixed_hamming_set.txt", sep=",", header=None)

    hamming_set = hamming_set.loc[:99]
    print("Ham Len", len(hamming_set))
    # print(hamming_set)

    fixed_array, moving_array, dvf_array = helper.shuffle_inplace(
        image_array, moving_array, dvf_array)

    # Ignore moving and dvf
    test_dataset, validation_moving, validation_dvf, trainVal_dataset, train_moving, train_dvf = helper.split_data(
        fixed_array, moving_array, dvf_array, split_ratio=0.05)
    validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
        trainVal_dataset, moving_array, dvf_array, split_ratio=0.15)

    normalised_train = helper.normalise(train_dataset)
    normalised_val = helper.normalise(validation_dataset)

    conc_data = np.concatenate((normalised_train,normalised_val))
    #prepare cross validation
    kf =KFold(n_splits=5,random_state=None, shuffle=True)

    kf.get_n_splits(conc_data)

    X = conc_data

    i = 1
    print(kf.split(X))

"""    for train_index, test_index in kf.split(X):
        K.clear_session()
        trainData = X[train_index]
        testData = X[test_index]


        train = X[0]
        test = X[1]
"""
    print("=========================================")
    print("====== K Fold Validation step => %d =======" % (i))
    print("=========================================")

    # Output all data from a training session into a dated folder
    outputPath = "./k-fold/k{}" .format(i)
    # hamming_list = [0, 1, 2, 3, 4]
    # img_idx = [0, 1, 2, 3, 4]
    # callbacks
    checkpoint = ModelCheckpoint(outputPath + '/best_model.h5', monitor='val_acc',
                                 verbose=1, save_best_only=True, period=1)
    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_acc', patience=10, verbose=1)
    # early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
    tensorboard = TrainValTensorBoard(write_graph=False, log_dir=outputPath)
    callbacks = [checkpoint, reduce_lr_plateau, tensorboard]
    # BUILD Model
    model = createSharedAlexnet3D_onemodel()
    # for layer in model.layers
    #     print(layer.name, layer.output_shape)
    opt = optimizers.SGD(lr=0.01)
    #plot_model(model, to_file='model.png')
    print(model.summary())
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(generator=gen.generator(train, list_avail_keys, hamming_set, batch_size=batch_size, N=num_permutations),
                        epochs=50, verbose=1,
                        steps_per_epoch=5,
                        validation_data=gen.generator(
        test, list_avail_keys, hamming_set, batch_size=batch_size, N=num_permutations),
        validation_steps=5, callbacks=callbacks, shuffle=False)
    model.save(outputPath + '/final_model{}.h5' .format(i))

    i+=1

def main(argv=None):
    train()


if __name__ == '__main__':
    main()
