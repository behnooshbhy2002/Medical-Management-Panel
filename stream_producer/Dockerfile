FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY producer.py .

# producer.py handles its own RabbitMQ retry loop,
# so no extra entrypoint wrapper is needed.
CMD ["python", "-u", "producer.py"]