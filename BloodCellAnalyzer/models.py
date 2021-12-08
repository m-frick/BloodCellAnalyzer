import numpy as np
import os
from tensorflow.keras.models import load_model
from BloodCellAnalyzer.data import img_seg


class malaria():

    def __init__(self):
        self.model = None
        self.X = 0

    def create_model(self, model):

        path = os.path.join(os.path.dirname(__file__), "../", "models/",
                            "malaria_cnn_models", str(model))

        model = load_model(path)

        for layer in model.layers:
            layer.trainable = False

        self.model = model


    def predict(self, path_X):

        self.X = img_seg(path_X)
        pred = self.model.predict(self.X)
        return pred
