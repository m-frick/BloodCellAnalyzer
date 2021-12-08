import numpy as np
import os
from tensorflow.keras.models import load_model
from BloodCellAnalyzer.rbc_seg import cell_crop


class Creator():

    def __init__(self):
        pass


    def load_malaria(self, model):
        """Loads pretrained model
            params{
            model str: either "leo" or "henry
            "henry takes grayscale as input, leo rgb"
        """

        path = os.path.join(os.path.dirname(__file__), "../",
                            "models/",
                            "malaria_cnn_models",
                            str(model))

        model = load_model(path)

        for layer in model.layers:
            layer.trainable = False

        return model
