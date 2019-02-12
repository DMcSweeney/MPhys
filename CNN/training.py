"""Training CNN for registration in Keras"""

from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, Conv3DTranspose
from keras.models import Model
from keras.initializers import RandomNormal
import dataLoader as load

fixed_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\PCT"
moving_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\PET"
dvf_dir = "E:\\MPhys\\DataSplit\\TrainingSet\\DVF"
# Load DATA
fixed_image, moving_image, dvf_label = load.data_reader(fixed_dir, moving_dir, dvf_dir)


fixed_array = fixed_image.get_data()
moving_array = moving_image.get_data()
dvf_array = dvf_label.get_data(is_image=False)


# Split into validation and training set
validation_fixed, validation_moving, validation_dvf, train_fixed, train_moving, train_dvf = load.split_data(
    fixed_array, moving_array, dvf_array, validation_ratio=0.2)

print("PCT Shape:", train_fixed.shape)
print("PET Shape:", train_moving.shape)
print("DVF Shape:", train_dvf.shape)

dvf_params = len(dvf_label.flatten[0])


# CNN Structure
fixed_image = Input(shape=(train_fixed.shape[1:]))  # Ignore batch but include channel
moving_image = Input(shape=(train_moving.shape[1:]))
input = concatenate([fixed_image, moving_image], axis=0)

x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='down_1')(input)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_1')(x1)

x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='down_2')(x)
x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='Pool_2')(x2)

x3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='down_3')(x)

x = UpSampling3D(size=(2, 2, 2), name='UpSamp_3')(x3)
y2 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same', name='Up_2')(x)
merge2 = concatenate([x2, y2], axis=0)

x = UpSampling3D(size=(2, 2, 2), name='UpSamp_2')(merge2)
y1 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same', name='Up_1')(x)

merge1 = concatenate([x1, y1], axis=0)

# flat = Flatten()(merge1)

# dense1 = Dense(1, activation='relu')(flat)
# dense2 = Dense(dvf_params, activation='softmax')(flat)

# Transform into flow field (from VoxelMorph Github)
dvf = Conv3D(3, kernel_size=3, padding='same', name='dvf',
             kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(merge1)

# Use merge 1 as input to DVF calc
# dvf =  # Some operation to get DVF
# Output DVF + Loss calc

model = Model(inputs=[fixed_image, moving_image], outputs=dvf)
for layer in model.layers:
    print(layer.output_shape)

print(model.summary())
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
model.fit([train_fixed, train_moving], [train_dvf], epochs=50, batch_size=4, verbose=1)
model.save('model.h5')
accuracy = model.evaluate(x=[validation_fixed, validation_moving],
                          y=validation_dvf, batch_size=4)
print("Accuracy:", accuracy[1])


# Train
