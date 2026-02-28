FROM python:3.11

WORKDIR /app

COPY consumer.py .

RUN pip install pika pymongo

CMD ["python", "consumer.py"]