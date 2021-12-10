import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model


class malaria():

    def __init__(self):
        self.model = None
        self.pred = None

        #ROI: Returns list of ROIs

    def load_malaria(self):
        """Loads pretrained model
            params{
            model str: either "leo" or "henry
            "henry takes grayscale as input, leo rgb"
        """
        self.model = None
        self.X = None
        self.img = None
        self.ROI = None
        self.ratio = None

        path = "models/malaria_cnn_models/henry"

        model = load_model(path)

        for layer in model.layers:
            layer.trainable = False

        self.model = model


    def predict(self, input_predict):

        pred = self.model.predict(input_predict)

        self.pred = np.round(pred, 2)
