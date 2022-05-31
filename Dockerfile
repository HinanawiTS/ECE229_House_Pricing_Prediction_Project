FROM python:3.9

WORKDIR /usr/src/app

COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


copy ./model.pkl ./
ADD ./data ./data
COPY ./application.py ./
ADD ./templates ./templates


EXPOSE 5000
CMD ["python", "application.py"]
