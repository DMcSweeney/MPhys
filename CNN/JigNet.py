from keras.layers import (Dense, Dropout, Concatenate, Input, Activation, Flatten, Conv3D,
                          MaxPooling3d, GlobalAveragePooling3D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import optimizers
from time import strftime, localtime
import dataLoader as load
import os
import dataGenerator as generator
import JigsawHelpers as help
#  from keras.utils import plot_model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

<<<<<<< HEAD
def basicModel():
=======

def train(tileSize=64, numPuzzles=9):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
    # Think we only need one directory since this uses both PET and PCT as fixed
    # moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"

    image_data, __image, __label = load.data_reader(image_dir, moving_dir, dvf_dir)

    image_array, image_affine = image_data.get_data()

    list_avail_keys = help.get_moveable_keys(image_array)
    # Get hamming set
    start_time = time.time()
    hamming_set = hamming.gen_max_hamming_set(num_permutations, avail_keys)
    end_time = time.time()
    print("Took {} to generate {} permutations". format(
        end_time - start_time, num_permutations))

    fixed_image = Input(shape=(image_array.shape[1:]))

    validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
        fixed_image, __image, __label, split_ratio=0.15)

    # CNN structure
>>>>>>> 49488d8fdd50c5f2641b88f0f0f7cff43c55fc5c
    modelInputs = Input(shape=(train_dataset.shape[1:]))

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

    model = Model(inputs=modelINputs, outputs=outputs  , name='Jigsaw_Model')

    return model

def trivialNet(tileSize=64, numPuzzles, hammingSetSize=10):

    inputShape = (tileSize, tileSize, 3)
    modelInputs = [Input(inputShape) for _ in range(numPuzzles)]
    sharedLayer = basicModel()
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]

    def L1_distance(x): return K.concatenate(
        [[K.abs(x[i] - x[j]) for j in range(i, numPuzzles )] for i in range(numPuzzles)])
    both = K.concatenate([[K.abs(x[0] - x[j]) for j in range(numPuzzles)])
    #  both = K.concatenate([[K.abs(x[i] - x[j]) for j in range(i, 9)] for i in range(9)])
                 #  output_shape=lambda x: x[0])

    x = Concatenate()(sharedLayers)  # Reconsider what axis to merge
    x = Dense(512, activation='relu')(x)
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


def train(tileSize=64, numPuzzles=9):
    # On server with PET and PCT in
    image_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/fixed"
    # Think we only need one directory since this uses both PET and PCT as fixed
    # moving_dir = "/hepgpu3-data1/dmcsween/DataTwoWay128/moving"

    image_data, __image, __label = load.data_reader(image_dir, moving_dir, dvf_dir)

    image_array, image_affine = image_data.get_data()

    fixed_image = Input(shape=(image_array.shape[1:]))

    validation_dataset, validation_moving, validation_dvf, train_dataset, train_moving, train_dvf = helper.split_data(
        fixed_image, __image, __label, split_ratio=0.15)



    dataGenerator = DataGenerator(batchSize=batch_size, meanTensor=normalize_mean,
                                  stdTensor=normalize_std, maxHammingSet=max_hamming_set[:hamming_set_size])

    # Output all data from a training session into a dated folder
    outputPath = ""
    os.makedirs(outputPath)

    # callbacks
    checkpointer = ModelCheckpoint(
        outputPath +
        '/weights_improvement.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True)
    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss', patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # BUILD Model
    model = trivialNet()
    

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)



    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=generator(image_array, list_avail_keys),
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
