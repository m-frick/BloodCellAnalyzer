FROM python:3.8.12-buster

COPY app /app
COPY requirements.txt /requirements.txt
COPY BloodCellAnalyzer /BloodCellAnalyzer
COPY models/malaria_cnn_models/henry /models/malaria_cnn_models/henry

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn app.fast:app --host 0.0.0.0  --port $PORT
