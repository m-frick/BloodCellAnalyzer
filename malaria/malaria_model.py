import numpy as np 
import pandas as pd
from tensorflow.keras.layers import Dropout , Dense, Activation, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_uniform


def create_model():
    model = Sequential()

    model.add(Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=glorot_uniform(),
            input_shape=(128, 128, 3))
            )
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="valid",
            kernel_initializer=glorot_uniform())
            )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            kernel_initializer=glorot_uniform(seed=2))
            )
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=glorot_uniform())
            )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="valid",
            kernel_initializer=glorot_uniform())
            )
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(60))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))

    return model
