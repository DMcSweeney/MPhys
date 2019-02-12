"""Training CNN for registration in Keras"""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, Conv3DTranspose
from keras.models import Model
import dataLoader as load

fixed_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\PCT"
moving_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\PET"
dvf_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\DVF"
# Load DATA
fixed_image, moving_image, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)

fixed_array = fixed_image.get_data()
moving_array = moving_image.get_data()
dvf_array = dvf_label.get_data(is_image=False)


print("PCT Shape:", fixed_array.shape)
print("PET Shape:", moving_array.shape)
print("DVF Shape:", dvf_array.shape)

# CNN Structure
fixed_image = Input(shape=(fixed_array.shape[1:]))  # Ignore batch but include channel
moving_image = Input(shape=(moving_array.shape[1:]))
input = concatenate([fixed_image, moving_image], axis=0)

x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x1)

x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x2)

x3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)

x = UpSampling3D(size=(2, 2, 2))(x3)
y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(x)
merge2 = concatenate([x2, y2], axis=0)

x = UpSampling3D(size=(2, 2, 2))(merge2)
y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(x)
merge1 = concatenate([x1, y1], axis=0)

# Use merge 1 as input to DVF calc
# dvf =  # Some operation to get DVF
# Output DVF + Loss calc

# input & output need to be keras tensors
model = Model(inputs=[fixed_image, moving_image], outputs=merge1)
print(model.summary())
