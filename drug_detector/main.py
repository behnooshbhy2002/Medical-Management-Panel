from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import os
import pika
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Drug Detector",
    description="Extracts mentioned medications from clinical text / prescriptions using spaCy NER + regex fallback",
    version="1.0.0"
)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

# ─────────────────────────────────────────
# Load spaCy medical model (optional)
# ─────────────────────────────────────────
USE_SPACY = False
nlp = None

try:
    import spacy
    nlp = spacy.load("en_core_sci_sm")
    USE_SPACY = True
    print("✅ spaCy medical model (en_core_sci_sm) loaded successfully")
except Exception:
    print("⚠️ spaCy not available — falling back to regex-only detection")
    USE_SPACY = False

# ─────────────────────────────────────────
# Common drugs list (regex fallback)
# ─────────────────────────────────────────
COMMON_DRUGS = [
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "amoxicillin",
    "metformin", "insulin", "lisinopril", "atorvastatin", "omeprazole",
    "amlodipine", "metoprolol", "simvastatin", "losartan", "albuterol",
    "claritin", "loratadine", "zyrtec", "cetirizine", "benadryl",
    "diphenhydramine", "prednisone", "dexamethasone", "nasonex", "fluticasone",
    "warfarin", "clopidogrel", "furosemide", "spironolactone", "hydrochlorothiazide",
    "gabapentin", "pregabalin", "sertraline", "fluoxetine", "escitalopram",
    "levothyroxine", "synthroid", "zithromax", "azithromycin", "doxycycline",
    "tri-cyclen", "norgestimate", "ethinyl", "estradiol", "tamoxifen",
    "ondansetron", "zofran", "pantoprazole", "esomeprazole", "ranitidine",
    # Add more brand/generic names as needed
]

DRUG_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(d) for d in COMMON_DRUGS) + r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str


# ─────────────────────────────────────────
# Core detection logic
# ─────────────────────────────────────────
def detect_drugs(text: str) -> dict:
    """
    Detect drug mentions using spaCy (if available) + regex fallback.
    Returns structured result with both simple list and detailed entities.
    """
    entities = []

    # 1. spaCy NER (preferred when available)
    if USE_SPACY and nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("CHEMICAL", "DRUG", "MEDICATION"):
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "method": "spacy-ner",
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

    # 2. Regex fallback / augmentation
    regex_matches = DRUG_PATTERN.finditer(text)
    found_texts = {e["text"].lower() for e in entities}

    for match in regex_matches:
        drug_name = match.group()
        if drug_name.lower() not in found_texts:
            entities.append({
                "text": drug_name,
                "label": "CHEMICAL",
                "method": "regex",
                "start": match.start(),
                "end": match.end(),
            })
            found_texts.add(drug_name.lower())

    # Format output
    detected_entities = {e["text"]: e["label"] for e in entities}

    return {
        "detected_entities": detected_entities,
        "drug_list": list(detected_entities.keys()),
        "drug_count": len(detected_entities),
        "method_used": "spacy+regex" if USE_SPACY else "regex-only",
        "detailed_entities": entities,
        "note": "This is automated extraction — verify clinically important medications"
    }


# ─────────────────────────────────────────
# RabbitMQ Background Consumer
# ─────────────────────────────────────────
def start_rabbitmq_consumer():
    """Background worker that processes prescription texts from queue"""
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
                    # Optional: publish result to another queue / save to DB
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(
            queue="prescription_queue",
            on_message_callback=callback
        )

        logger.info("RabbitMQ consumer started – listening on 'prescription_queue'")
        channel.start_consuming()

    except Exception as e:
        logger.error(f"RabbitMQ consumer failed: {e}")


@app.on_event("startup")
def startup_event():
    """Launch RabbitMQ consumer thread at startup"""
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
        "status": "operational"
    }


@app.post("/detect")
def detect_drugs_endpoint(data: TextInput):
    """
    Extract mentioned medications from clinical/prescription text
    """
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return detect_drugs(data.text)


@app.get("/drug-list")
def get_common_drug_list():
    """Return the list of drugs the regex pattern is looking for"""
    return {
        "common_drugs": COMMON_DRUGS,
        "count": len(COMMON_DRUGS),
        "note": "This is the static keyword-based list used in regex fallback"
    }


@app.get("/health")
def health_check():
    """Health check for container orchestrators"""
    return {
        "status": "ok",
        "spacy_loaded": USE_SPACY,
        "regex_drug_count": len(COMMON_DRUGS),
        "rabbitmq_configured": bool(RABBITMQ_URL)
    }