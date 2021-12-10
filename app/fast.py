from sys import path
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from BloodCellAnalyzer.rbc_seg import cell_crop, rbc
from BloodCellAnalyzer.models import Creator
from BloodCellAnalyzer.data import img_seg



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"])  # Allows all headers


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.post("/segmenter")
async def segmenter(file: UploadFile = File(...)):

    image = np.array(Image.open(file.file))

    list_ROI = cell_crop(image)

    input_predict = img_seg(list_ROI[1:])

    creator = Creator()

    creator.load_malaria()

    creator.predict(input_predict)

    pred = creator.pred.tolist()


    #print("image: ", file)

    #Image.open(file.file).show()

    return {"list_ROI": str(list_ROI), "predictions": str(pred)}
