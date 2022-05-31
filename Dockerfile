FROM python:3.9

WORKDIR /usr/src/app

COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


ADD ./model.pkl ./model.pkl
ADD ./data ./data
COPY ./application.py ./
ADD ./templates ./templates
ADD ./pred ./pred
ADD ./static ./static
ADD ./viz ./viz
ADD ./viz_FilterbyText ./viz_FilterbyText


EXPOSE 5000
CMD ["python", "application.py"]
