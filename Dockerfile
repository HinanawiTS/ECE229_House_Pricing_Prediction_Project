FROM python:3.9

WORKDIR /usr/src/app

COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY ./model.pkl ./
COPY ./data ./
COPY ./application.py ./
COPY ./templates ./
COPY ./predict.py ./
COPY ./pred ./
COPY ./static ./
COPY ./viz ./
COPY ./viz_FilterbyText ./
COPY ./data_preprocess.ipynb ./
COPY ./model_fake.pkl ./

EXPOSE 5000
RUN python application.py
