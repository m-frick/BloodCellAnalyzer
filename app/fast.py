from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from BloodCellAnalyzer.rbc_seg import cell_crop, rbc


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

    print("image: ", file)

    Image.open(file.file).show()

    return {"status": "uploaded" }
