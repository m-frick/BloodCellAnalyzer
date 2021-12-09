import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from BloodCellAnalyzer.data import img_seg
from BloodCellAnalyzer.rbc_seg import cell_crop


class malaria():

    def __init__(self):
        """
        Attributes of malaria_class:

        X: Returns nd_array of grayscale ROIs

        img: Return input image

        ROI: Returns list of ROIs

        ratio: Returns ratio if infected cells to infected cells
        """
        self.model = None
        self.X = None
        self.img = None
        self.ROI = None
        self.ratio = None

    def create_model(self, model):

        path = os.path.join(os.path.dirname(__file__), "../", "models/",
                            "malaria_cnn_models", str(model))

        model = load_model(path)

        for layer in model.layers:
            layer.trainable = False

        self.model = model


    def predict(self, path_X):

        cells = cell_crop(path_X)

        self.img = cells[0]

        self.ROI = cells[1:]

        self.X = img_seg(self.ROI)

        pred = self.model.predict(self.X)

        res = np.array(["Uninfected" if i > 0.5 else "Infected" for i in pred])

        np.set_printoptions(suppress=True)

        pred = np.squeeze(pred)

        df = pd.DataFrame({"Proba of Inf(%)": pred.round(2), "Status": res})

        self.ratio = df[df["Proba of Inf(%)"] < 0.5].count()[0] / (
            len(df) - df[df["Proba of Inf(%)"] < 0.5].count()[0])

        return df
