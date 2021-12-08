import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


class Creator():

    def __init__(self):
        pass


    def load_malaria(self):
        "Loads pretrained model"

        path = os.path.join(os.path.dirname(__file__), "../", "models/", "malaria_cnn_model")

        model = load_model(path)

        for layer in model.layers:
            layer.trainable = False

        return model
