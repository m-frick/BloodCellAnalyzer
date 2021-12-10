import numpy as np
import os
from tensorflow.keras.models import load_model


class Creator():

    def __init__(self):
        self.model = None
        self.pred = None


    def load_malaria(self):
        """Loads pretrained model
            params{
            model str: either "leo" or "henry
            "henry takes grayscale as input, leo rgb"
        """

        path = "models/malaria_cnn_models/henry"

        model = load_model(path)

        for layer in model.layers:
            layer.trainable = False

        self.model = model


    def predict(self, input_predict):

        pred = self.model.predict(input_predict)

        self.pred = np.round(pred, 2)
