from keras.layers import (Dense, Concatenate, Input, Activation, Flatten,
                          Conv3D, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras import optimizers
from keras.utils import plot_model
import dataLoader as load
from customTensorBoard import TrainValTensorBoard
import dataGenerator as gen
import JigsawHelpers as help
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
#  from keras.utils import plot_model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def createSharedAlexnet3D(tileSize=25, numPuzzles=23, hammingSetSize=25):

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


def createShortSharedAlexnet3D(tileSize=25, numPuzzles=23, hammingSetSize=25):

    input_layers = [Input(shape=(tileSize, tileSize, tileSize, 1),
                          name="alexnet_input_{}".format(n)) for n in range(numPuzzles)]
    conv1 = Conv3D(96, (3, 3, 3), strides=(4, 4, 4), activation='relu',
                   padding='valid', name="Convolutional_1")
    bn1 = BatchNormalization(name="BatchNorm_1")

    conv2 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), activation='relu',
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


def train(tileSize=64, numPuzzles=23, num_permutations=25, batch_size=32):
    # Seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
    print("Load Data")
    image_data, __image, __label = load.data_reader(image_dir, image_dir, image_dir)

    image_array, image_affine = image_data.get_data()
    moving_array, moving_affine = __image.get_data()
    dvf_array, dvf_affine = __label.get_data()

    list_avail_keys = help.get_moveable_keys(image_array)
    # Get hamming set
    print("Load hamming Set")
    hamming_set = pd.read_csv("hamming_set.txt", sep=",", header=None)
    print(hamming_set)
    # Ignore moving and dvf
    # validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
    #    image_array, moving_array, dvf_array, split_ratio=0.15)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # Cross val scores
    cvscores = []
    # Output all data from a training session into a dated folder
    outputPath = "./logs"
    # callbacks
    checkpointer = ModelCheckpoint(outputPath + '/weights_improvement.hdf5',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True, period=1)
    # reduce_lr_plateau = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1)
    # early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
    tensorboard = TrainValTensorBoard(write_graph=False)
    callbacks = [checkpointer, tensorboard]
    # BUILD Model
    n_samples = image_array.shape[0]
    for train, val in kfold.split(np.zeros(n_samples), np.zeros(n_samples)):
        print(train, val)
        model = createSharedAlexnet3D(hammingSetSize=10)
        """
        for layer in model.layers:
            print(layer.name, layer.output_shape)
        plot_model(model, to_file='model.png')
        print(model.summary())
        """
        opt = optimizers.Adam(lr=0.1)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(generator=gen.generator(image_array[train], list_avail_keys, hamming_set, batch_size=batch_size, N=10),
                            epochs=25, verbose=2,
                            steps_per_epoch=len(train) // batch_size, callbacks=callbacks)
        scores = model.evaluate_generator(gen.generator(image_array[val], list_avail_keys, hamming_set, batch_size=4, N=10),
                                          steps=len(val) // batch_size)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    model.save('model.h5')


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
