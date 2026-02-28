from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import pika
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical disease Classifier",
    description="Classifies medical disease from prescription or clinical text using ModernBERT",
    version="1.1.0"
)

# ─────────────────────────────────────────
# Configuration & Model Loading
# ─────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/disease")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

# Global variables
tokenizer = None
model = None
classes = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256   # دقیقاً مثل مرحله آموزش

try:
    model_path = os.path.join(MODEL_DIR, "specialty_modernbert")
    classes_path = os.path.join(MODEL_DIR, "specialty_classes.pkl")

    logger.info(f"🔄 Loading ModernBERT model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(DEVICE)
    model.eval()

    classes = joblib.load(classes_path)

    logger.info(f"✅ ModernBERT model loaded successfully! "
                f"({len(classes)} classes) | Device: {DEVICE}")
except Exception as e:
    logger.error(f"⚠️ Failed to load model: {e}")
    tokenizer = model = classes = None


# ─────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str


# ─────────────────────────────────────────
# Prediction Function (ModernBERT)
# ─────────────────────────────────────────
def classify(text: str) -> dict:
    """Run inference with ModernBERT and return top predictions"""
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    if not text or not text.strip():
        return {"error": "Empty text"}

    # پیش‌پردازش سبک (دقیقاً مثل مرحله آموزش)
    cleaned = text.replace("User:", "").strip()

    # Tokenization
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1)

    # Top-3
    top3_probs, top3_idx = torch.topk(probs, 3)

    return {
        "predicted_disease": classes[top3_idx[0].item()],
        "confidence": round(float(top3_probs[0].item()), 4),
        "top3": [
            {
                "disease": classes[idx.item()],
                "confidence": round(float(prob.item()), 4)
            }
            for prob, idx in zip(top3_probs, top3_idx)
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
                    disease = result.get("predicted_disease", "unknown")
                    logger.info(f"[RabbitMQ] Processed prescription → {disease} "
                                f"(conf: {result.get('confidence', 0)})")
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing RabbitMQ message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(
            queue="prescription_queue",
            on_message_callback=callback
        )

        logger.info("🐰 RabbitMQ consumer started – listening on 'prescription_queue'")
        channel.start_consuming()

    except Exception as e:
        logger.error(f"RabbitMQ consumer failed to start: {e}")


@app.on_event("startup")
def startup_event():
    """Start RabbitMQ consumer in background thread"""
    if model is None:
        logger.warning("Model not loaded – RabbitMQ consumer will not process messages")
        return

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
        "service": "Medical disease Classifier (ModernBERT)",
        "status": "operational",
        "model": "answerdotai/ModernBERT-base (fine-tuned)"
    }


@app.post("/classify")
def classify_text(data: TextInput):
    """
    Classify medical disease from input text (prescription, note, conversation, etc.)
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
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": len(classes) if classes is not None else 0,
        "rabbitmq_configured": bool(RABBITMQ_URL),
        "model_type": "ModernBERT (HuggingFace)"
    }