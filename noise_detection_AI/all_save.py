import plaidml.keras
plaidml.keras.install_backend()
import pandas as pd
import os
import sys

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint


base_path = 'D:/Users/pr0gr/RD_club/projects/G2021-maikataA_py/noise_detection_AI/ESC-50'
path = os.path.join(base_path, 'meta/esc50.csv')
csv_input = pd.read_csv(path, encoding="ms932", sep=",")

path = []

for i in range(2000):
    if csv_input.category[i] == 'snoring':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
        print(path)
    elif csv_input.category[i] == 'washing_machine':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
    elif csv_input.category[i] == 'vacuum_cleaner':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
    elif csv_input.category[i] == 'helicopter':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
    elif csv_input.category[i] == 'chainsaw':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
    elif csv_input.category[i] == 'engine':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
    elif csv_input.category[i] == 'airplane':
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])
    else:
        path[i] = os.path.join(base_path, 'audio' + csv_input.filename[i])


sys.exit()
# redefine target data into one hot vector
classes = 50
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# define CNN
inputs = Input(shape=(x_train.shape[1:]))

x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))

x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
x_2 = cba(x_2, filters=64, kernel_size=(16,1), strides=(2,1))

x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
x_3 = cba(x_3, filters=64, kernel_size=(32,1), strides=(2,1))

x_4 = cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
x_4 = cba(x_4, filters=64, kernel_size=(64,1), strides=(2,1))

x = Add()([x_1, x_2, x_3, x_4])

x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

x = GlobalAveragePooling2D()(x)
x = Dense(classes)(x)
x = Activation("softmax")(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()
