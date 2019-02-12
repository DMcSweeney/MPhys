"""Training CNN for registration in Keras"""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, Conv3DTranspose
from keras.models import Model


# Load DATA


# CNN Structure
fixed_image = Input(shape=image_shape)  # Change shape
moving_image = Input(shape=image_shape)
input = concatenate([fixed_image, moving_image])

x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x1)

x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x2)

x3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)

x = UpSampling3D(size=(2, 2, 2))(x3)
y3 = Conv3DTranspose(256, (3, 3, 3), activation='relu', padding='same')(x)
merge3 = concatenate([x3, y3], axis=-1)

x = UpSampling3D(size=(2, 2, 2))(merge3)
y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(x)
merge2 = concatenate([x2, y2], axis=-1)

x = UpSampling3D(size=(2, 2, 2))(y2)
y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(x)
merge1 = concatenate([x1, y1], axis=-1)

# Use merge 1 as input to DVF calc
dvf =  # Some operation to get DVF
# Output DVF + Loss calc
model = Model(input=[fixed_image, moving_image], outputs=dvf)
# Train
