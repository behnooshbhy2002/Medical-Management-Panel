from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import re
import pika
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Specialty Classifier",
    description="Classifies medical specialty from prescription or clinical text using a trained model",
    version="1.0.0"
)

# ─────────────────────────────────────────
# Configuration & Model Loading
# ─────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

pipeline = None
le = None
classes = None

try:
    pipeline = joblib.load(os.path.join(MODEL_DIR, "specialty_pipeline.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "specialty_label_encoder.pkl"))
    classes = joblib.load(os.path.join(MODEL_DIR, "specialty_classes.pkl"))
    print(f"✅ Specialty classification model loaded ({len(classes)} classes)")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    pipeline = le = classes = None


# ─────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str


# ─────────────────────────────────────────
# Text Preprocessing & Prediction
# ─────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove numbers & punctuation, normalize spaces"""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)               # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)           # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()       # normalize spaces
    return text


def classify(text: str) -> dict:
    """Run inference and return top predictions"""
    if pipeline is None:
        return {"error": "Model not loaded"}

    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "Empty text after cleaning"}

    probs = pipeline.predict_proba([cleaned])[0]
    top3_idx = probs.argsort()[-3:][::-1]

    return {
        "predicted_specialty": classes[top3_idx[0]],
        "confidence": round(float(probs[top3_idx[0]]), 4),
        "top3": [
            {"specialty": classes[i], "confidence": round(float(probs[i]), 4)}
            for i in top3_idx
        ],
        "note": "This is a model prediction — clinical judgment should prevail"
    }


# ─────────────────────────────────────────
# RabbitMQ Background Consumer
# ─────────────────────────────────────────
def start_rabbitmq_consumer():
    """Background worker that consumes prescription texts from RabbitMQ"""
    try:
        conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = conn.channel()
        channel.queue_declare(queue="prescription_queue", durable=True)

        def callback(ch, method, properties, body):
            try:
                data = json.loads(body)
                text = data.get("text", "")
                if text:
                    result = classify(text)
                    specialty = result.get("predicted_specialty", "unknown")
                    logger.info(f"[RabbitMQ] Processed prescription → {specialty}")
                    # Here you could publish result somewhere else if needed
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing RabbitMQ message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(
            queue="prescription_queue",
            on_message_callback=callback
        )

        logger.info("RabbitMQ consumer started – listening on 'prescription_queue'")
        channel.start_consuming()

    except Exception as e:
        logger.error(f"RabbitMQ consumer failed to start: {e}")


@app.on_event("startup")
def startup_event():
    """Start RabbitMQ consumer in background thread on application startup"""
    consumer_thread = threading.Thread(
        target=start_rabbitmq_consumer,
        daemon=True,
        name="RabbitMQ-Consumer"
    )
    consumer_thread.start()


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Medical Specialty Classifier",
        "status": "operational"
    }


@app.post("/classify")
def classify_text(data: TextInput):
    """
    Classify medical specialty from input text (prescription, note, etc.)
    Returns top prediction + top-3 probabilities
    """
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return classify(data.text)


@app.get("/classes")
def list_supported_classes():
    """Return list of all supported medical specialties"""
    if classes is None:
        return {"classes": [], "note": "Model not loaded"}
    return {"classes": list(classes)}


@app.get("/health")
def health_check():
    """Health check endpoint for orchestrators (Kubernetes, Docker Compose, etc.)"""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "num_classes": len(classes) if classes is not None else 0,
        "rabbitmq_configured": bool(RABBITMQ_URL)
    }