"""
DrugBank Biotech Drugs Consumer Microservice
- Consumes messages from RabbitMQ
- Stores / updates documents in MongoDB
"""

import json
import logging
import pika
from pymongo import MongoClient

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

RABBITMQ_URL = "amqp://guest:guest@rabbitmq-service/"
QUEUE_NAME   = "drugbank_biotech_drugs"

MONGO_URI       = "mongodb://mongodb:27017/"
DB_NAME         = "drugbank"
COLLECTION_NAME = "biotech_drugs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

collection = None  # global reference for callback


def callback(ch, method, properties, body):
    global collection
    try:
        drug = json.loads(body.decode("utf-8"))
        
        result = collection.update_one(
            {"link": drug.get("link")},
            {"$set": drug},
            upsert=True
        )
        
        if result.upserted_id:
            log.info(f"✅ New drug inserted: {drug.get('name', 'N/A')}")
        else:
            log.info(f"🔄 Drug updated: {drug.get('name', 'N/A')}")
            
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        log.error(f"❌ Error processing message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


if __name__ == "__main__":
    # Connect to MongoDB
    client = None
    try:
        log.info("🌍 Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # force connection test
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        collection.create_index([("link", 1)], unique=True)  # prevent duplicates
        log.info(f"✅ MongoDB ready — {DB_NAME}.{COLLECTION_NAME}")
    except Exception as e:
        log.error(f"Failed to connect to MongoDB: {e}")
        exit(1)

    # RabbitMQ Consumer
    connection = None
    try:
        log.info("🐰 Connecting to RabbitMQ...")
        connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)
        
        log.info(f"🚀 Consumer started — consuming from queue '{QUEUE_NAME}'...")
        log.info("Press Ctrl+C to exit")
        channel.start_consuming()
    except KeyboardInterrupt:
        log.info("⛔ Consumer stopped by user.")
    except Exception as e:
        log.error(f"General error: {e}")
    finally:
        if connection and not connection.is_closed:
            connection.close()
        if client:
            client.close()
        log.info("🏁 Consumer shutdown complete.")