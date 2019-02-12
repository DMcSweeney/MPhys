"""Training CNN for registration in Keras"""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, Conv3DTranspose
from keras.models import Model
<<<<<<< HEAD
import dataLoader as load
=======
from keras.layers import Dense
from keras.layers import Flatten
import glob
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
>>>>>>> fa700fb92b38c86cb71c73e953ac0869fa6174e8

fixed_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\PCT"
moving_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\PET"
dvf_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\DVF"
# Load DATA
fixed_image, moving_image, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)

<<<<<<< HEAD
fixed_array = fixed_image.get_data()
moving_array = moving_image.get_data()
dvf_array = dvf_label.get_data(is_image=False)
=======
for filename in glob.glob('D:\\Mphys\\Nifty\\PET\\*.nii'):
    inputPET = nib.load(filename)
    inputPETNumpy = np.array(inputPET.dataobj)
    PETList.append(inputPETNumpy)
    PETArray = np.asarray(PETList)

print(PETArray.shape)


PCTList = []

for filename in glob.glob('D:\\Mphys\\Nifty\\PCT\\*.nii'):
    inputPCT = nib.load(filename)
    inputPCTNumpy = np.array(inputPCT.dataobj)
    PCTList.append(inputPCTNumpy)
    PCTArray = np.asarray(PCTList)


print(PCTArray.shape)


DVFList = []

for filename in glob.glob('D:\\Mphys\\Nifty\\DVF\\*.nii'):
    inputDVF = nib.load(filename)
    inputDVFNumpy = np.array(inputDVF.dataobj)
    DVFList.append(inputDVFNumpy)
    DVFArray = np.asarray(DVFList)

print(DVFArray.shape)
>>>>>>> fa700fb92b38c86cb71c73e953ac0869fa6174e8


print("PCT Shape:", fixed_array.shape)
print("PET Shape:", moving_array.shape)
print("DVF Shape:", dvf_array.shape)

<<<<<<< HEAD
# CNN Structure
fixed_image = Input(shape=(fixed_array.shape[1:]))  # Ignore batch but include channel
moving_image = Input(shape=(moving_array.shape[1:]))
input = concatenate([fixed_image, moving_image], axis=0)
=======
#] CNN Structure
fixed_image = Input(shape=PETArray.shape)  # Change shape
moving_image = Input(shape=PCTArray.shape)
input = concatenate([fixed_image, moving_image])
>>>>>>> fa700fb92b38c86cb71c73e953ac0869fa6174e8

x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x1)

x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x2)

x3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)

x = UpSampling3D(size=(2, 2, 2))(x3)
<<<<<<< HEAD
y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(x)
merge2 = concatenate([x2, y2], axis=0)
=======
'''y3 = Conv3DTranspose(256, (3, 3, 3), activation='relu', padding='same')(x)
merge3 = concatenate([x3, y3], axis=-1)

x = UpSampling3D(size=(2, 2, 2))(merge3)'''
y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(x)
merge2 = concatenate([x2, y2], axis=1)
>>>>>>> fa700fb92b38c86cb71c73e953ac0869fa6174e8

x = UpSampling3D(size=(2, 2, 2))(merge2)
y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(x)
<<<<<<< HEAD
merge1 = concatenate([x1, y1], axis=0)
=======

merge1 = concatenate([x1, y1], axis=1)

flat = Flatten()(merge1)

dense1 = Dense(1, activation='relu')(flat)
dense2 = Dense(786432,activation = 'softmax')(dense1)

>>>>>>> fa700fb92b38c86cb71c73e953ac0869fa6174e8

# Use merge 1 as input to DVF calc
# dvf =  # Some operation to get DVF
# Output DVF + Loss calc
<<<<<<< HEAD

# input & output need to be keras tensors
model = Model(inputs=[fixed_image, moving_image], outputs=merge1)
print(model.summary())
=======
model = Model(input=[fixed_image, moving_image], outputs=dense2)
print(model.summary())
# Train
>>>>>>> fa700fb92b38c86cb71c73e953ac0869fa6174e8
