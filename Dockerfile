FROM python:3.10

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt 

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "topic_clustering/main_pipeline.py"]