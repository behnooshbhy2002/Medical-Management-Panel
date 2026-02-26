# =============================================================
# FILE: stream_producer/producer.py
# ROLE: Reads rows from CSV files every 5 seconds and
#       publishes them to the appropriate RabbitMQ queues.
#
# Queues published:
#   - queue.diabetes        <- diabetes_prediction_dataset.csv
#   - queue.disease         <- medical_conversations.csv
#   - queue.drug            <- mtsamples.csv
# =============================================================

import pika
import pandas as pd
import json
import time
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

DATA_DIR = os.getenv("DATA_DIR", "/app/data")

DIABETES_CSV = os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv")
DISEASE_CSV  = os.path.join(DATA_DIR, "medical_conversations.csv")
DRUG_CSV     = os.path.join(DATA_DIR, "mtsamples.csv")

PUBLISH_INTERVAL = int(os.getenv("PUBLISH_INTERVAL", "5"))   # seconds between each row

QUEUE_DIABETES = "queue.diabetes"
QUEUE_DISEASE  = "queue.disease"
QUEUE_DRUG     = "queue.drug"

# ── Load CSVs ─────────────────────────────────────────────────
def load_csv(path: str, required_cols: list[str]) -> pd.DataFrame | None:
    if not os.path.exists(path):
        logger.warning(f"CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns {missing} in {path}")
        return None
    df = df.dropna(subset=required_cols)
    logger.info(f"Loaded {len(df)} rows from {os.path.basename(path)}")
    return df

def load_all() -> dict:
    diabetes_df = load_csv(DIABETES_CSV, [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "hba1c_level", "blood_glucose_level",
    ])
    disease_df = load_csv(DISEASE_CSV, ["conversations", "disease"])
    drug_df    = load_csv(DRUG_CSV,    ["transcription"])
    return {
        QUEUE_DIABETES: diabetes_df,
        QUEUE_DISEASE:  disease_df,
        QUEUE_DRUG:     drug_df,
    }

# ── RabbitMQ connection (with retry) ─────────────────────────
def connect_rabbitmq(retries: int = 10, delay: int = 5) -> pika.BlockingConnection:
    for attempt in range(1, retries + 1):
        try:
            conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            logger.info("Connected to RabbitMQ")
            return conn
        except Exception as e:
            logger.warning(f"Connection attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to RabbitMQ after retries")

def declare_queues(channel: pika.adapters.blocking_connection.BlockingChannel):
    for q in (QUEUE_DIABETES, QUEUE_DISEASE, QUEUE_DRUG):
        channel.queue_declare(queue=q, durable=True)
    logger.info("All queues declared")

# ── Row serializers ───────────────────────────────────────────
def serialize_diabetes(row: pd.Series) -> dict:
    smoking_vals = ["never", "no info", "former", "current", "ever", "not current"]
    smoking = str(row.get("smoking_history", "No Info"))
    if smoking.lower() not in smoking_vals:
        smoking = "No Info"
    return {
        "gender":             str(row.get("gender", "Male")),
        "age":                float(row.get("age", 40)),
        "hypertension":       int(row.get("hypertension", 0)),
        "heart_disease":      int(row.get("heart_disease", 0)),
        "smoking_history":    smoking,
        "bmi":                round(float(row.get("bmi", 25.0)), 1),
        "hba1c_level":        round(float(row.get("hba1c_level", 5.5)), 1),
        "blood_glucose_level":round(float(row.get("blood_glucose_level", 100)), 1),
    }

def serialize_disease(row: pd.Series) -> dict:
    text = str(row.get("conversations", "")).replace("User:", "").strip()
    return {
        "text":          text,
        "ground_truth":  str(row.get("disease", "")),
    }

def serialize_drug(row: pd.Series) -> dict:
    text = str(row.get("transcription", "")).strip()
    return {
        "text":             text[:1000],   # cap length
        "medical_specialty":str(row.get("medical_specialty", "")),
    }

SERIALIZERS = {
    QUEUE_DIABETES: serialize_diabetes,
    QUEUE_DISEASE:  serialize_disease,
    QUEUE_DRUG:     serialize_drug,
}

# ── Publish one message ───────────────────────────────────────
def publish(channel, queue: str, payload: dict):
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(payload),
        properties=pika.BasicProperties(
            delivery_mode=2,          # persistent
            content_type="application/json",
        ),
    )

# ── Main loop ─────────────────────────────────────────────────
def main():
    logger.info("Stream producer starting...")

    dataframes = load_all()

    # Check at least one dataset loaded
    if all(df is None for df in dataframes.values()):
        raise RuntimeError("No CSV files loaded. Check DATA_DIR and file names.")

    conn    = connect_rabbitmq()
    channel = conn.channel()
    declare_queues(channel)

    # Maintain per-queue index cursors (cycle through rows indefinitely)
    cursors = {q: 0 for q in dataframes}

    logger.info(f"Publishing every {PUBLISH_INTERVAL}s. Press Ctrl+C to stop.")

    while True:
        for queue, df in dataframes.items():
            if df is None or len(df) == 0:
                continue

            idx = cursors[queue] % len(df)
            row = df.iloc[idx]

            try:
                payload = SERIALIZERS[queue](row)
                publish(channel, queue, payload)
                logger.info(
                    f"[{queue}] row {idx+1}/{len(df)} published"
                )
            except Exception as e:
                logger.error(f"[{queue}] Failed to publish row {idx}: {e}")
                # Reconnect on channel errors
                try:
                    conn    = connect_rabbitmq(retries=3, delay=2)
                    channel = conn.channel()
                    declare_queues(channel)
                except Exception:
                    pass

            cursors[queue] += 1

        time.sleep(PUBLISH_INTERVAL)


if __name__ == "__main__":
    main()