from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pika
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Drug Detector",
    description="Extracts medications (CHEMICAL entities) from clinical text using SciSpacy en_ner_bc5cdr_md",
    version="1.1.0"
)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

# ─────────────────────────────────────────
# Load SciSpacy BC5CDR model (only CHEMICAL)
# ─────────────────────────────────────────
nlp = None
MODEL_NAME = "en_ner_bc5cdr_md"

try:
    import spacy
    nlp = spacy.load(MODEL_NAME)
    logger.info(f"✅ SciSpacy model '{MODEL_NAME}' loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load {MODEL_NAME}: {e}")
    raise RuntimeError("SciSpacy model is required!") from e

# ─────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str


# ─────────────────────────────────────────
# Core detection logic - ONLY SciSpacy CHEMICAL
# ─────────────────────────────────────────
def detect_drugs(text: str) -> dict:
    """
    Detect drug mentions using ONLY en_ner_bc5cdr_md model (CHEMICAL entities).
    """
    if not text or not text.strip():
        return {
            "detected_entities": {},
            "drug_list": [],
            "drug_count": 0,
            "method_used": MODEL_NAME,
            "detailed_entities": [],
            "note": "Empty text"
        }

    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ == "CHEMICAL":
            drug_text = ent.text.strip()
            if drug_text.lower() not in seen:
                seen.add(drug_text.lower())
                entities.append({
                    "text": drug_text,
                    "label": ent.label_,
                    "method": "scispacy_bc5cdr",
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

    drug_list = [e["text"] for e in entities]

    return {
        "detected_entities": {e["text"]: e["label"] for e in entities},
        "drug_list": drug_list,
        "drug_count": len(drug_list),
        "method_used": MODEL_NAME,
        "detailed_entities": entities,
        "note": "Powered by SciSpacy BC5CDR model — Only CHEMICAL entities returned as medications. Always verify clinically."
    }


# ─────────────────────────────────────────
# RabbitMQ Background Consumer (unchanged)
# ─────────────────────────────────────────
def start_rabbitmq_consumer():
    try:
        conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = conn.channel()
        channel.queue_declare(queue="prescription_queue", durable=True)

        def callback(ch, method, properties, body):
            try:
                data = json.loads(body)
                text = data.get("text", "")
                if text.strip():
                    result = detect_drugs(text)
                    count = result["drug_count"]
                    logger.info(f"[RabbitMQ] Detected {count} drug mention(s)")
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue="prescription_queue", on_message_callback=callback)
        logger.info("RabbitMQ consumer started – listening on 'prescription_queue'")
        channel.start_consuming()

    except Exception as e:
        logger.error(f"RabbitMQ consumer failed: {e}")


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(
        target=start_rabbitmq_consumer,
        daemon=True,
        name="DrugDetector-RabbitMQ-Consumer"
    )
    thread.start()


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Clinical Drug Detector",
        "model": MODEL_NAME,
        "status": "operational"
    }


@app.post("/detect")
def detect_drugs_endpoint(data: TextInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return detect_drugs(data.text)


@app.get("/model-info")
def get_model_info():
    """اطلاعات مدل استفاده‌شده"""
    return {
        "model": MODEL_NAME,
        "description": "BioCreative V CDR corpus — Chemical & Disease NER",
        "entities_used": ["CHEMICAL"],
        "note": "Only CHEMICAL entities are returned as medications"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": nlp is not None,
        "model_name": MODEL_NAME,
        "rabbitmq_configured": bool(RABBITMQ_URL)
    }