"""
API Gateway — Entry point for all incoming requests
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
import pika
import json
import os
import asyncio
from contextlib import asynccontextmanager
from databases import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:secret@postgres/health_db")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

SERVICES = {
    "diabetes_predictor": os.getenv("DIABETES_PREDICTOR_URL", "http://diabetes_predictor:8000"),
    "diabetes_stats": os.getenv("DIABETES_STATS_URL", "http://diabetes_stats:8000"),
    "specialty_classifier": os.getenv("SPECIALTY_CLASSIFIER_URL", "http://specialty_classifier:8000"),
    "drug_detector": os.getenv("DRUG_DETECTOR_URL", "http://drug_detector:8000"),
}

database = Database(DATABASE_URL)

# ─────────────────────────────────────────
# Input Schemas
# ─────────────────────────────────────────
class PatientData(BaseModel):
    gender: str = Field(..., example="Male")
    age: float = Field(..., ge=0, le=120, example=45.0)
    hypertension: int = Field(..., ge=0, le=1, example=0)
    heart_disease: int = Field(..., ge=0, le=1, example=0)
    smoking_history: str = Field(..., example="never")
    bmi: float = Field(..., ge=10, le=70, example=27.5)
    hba1c_level: float = Field(..., ge=3, le=15, example=5.8)
    blood_glucose_level: float = Field(..., ge=50, le=400, example=120.0)


class PrescriptionData(BaseModel):
    text: str = Field(..., min_length=10, example="Patient prescribed Claritin 10mg for allergic rhinitis...")


# ─────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    await create_tables()
    logger.info("✅ Database connected")
    yield
    await database.disconnect()
    logger.info("Database disconnected")


app = FastAPI(
    title="Health AI — API Gateway",
    description="API Gateway for the intelligent diabetes risk analysis system",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────
# Create tables
# ─────────────────────────────────────────
async def create_tables():
    await database.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            type VARCHAR(50) NOT NULL,
            input_data JSONB,
            result JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)


# ─────────────────────────────────────────
# Publish message to RabbitMQ
# ─────────────────────────────────────────
def publish_to_rabbitmq(queue: str, message: dict):
    try:
        conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = conn.channel()
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)  # persistent
        )
        conn.close()
        logger.info(f"📨 Message published to queue: {queue}")
    except Exception as e:
        logger.error(f"RabbitMQ error: {e}")


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "API Gateway is running", "services": list(SERVICES.keys())}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "database": "connected"}


@app.post("/predict/diabetes", tags=["Diabetes"])
async def predict_diabetes(data: PatientData):
    """Predict diabetes risk for a patient"""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{SERVICES['diabetes_predictor']}/predict",
                json=data.model_dump()
            )
            resp.raise_for_status()
            result = resp.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

    # Save to database
    await database.execute(
        "INSERT INTO predictions (type, input_data, result) VALUES (:t, :i, :r)",
        {"t": "diabetes", "i": json.dumps(data.model_dump()), "r": json.dumps(result)}
    )

    return result


@app.get("/stats/diabetes", tags=["Diabetes"])
async def diabetes_stats():
    """Get diabetes statistics report"""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{SERVICES['diabetes_stats']}/stats")
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@app.post("/analyze/prescription", tags=["Prescription"])
async def analyze_prescription(data: PrescriptionData):
    """
    Analyze medical prescription:
    - Classify medical specialty
    - Extract medications
    (using both direct HTTP and RabbitMQ approaches)
    """
    # Publish to queue for streaming/async processing
    publish_to_rabbitmq("prescription_queue", {"text": data.text})

    # Direct HTTP calls (synchronous result)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            specialty_task = client.post(
                f"{SERVICES['specialty_classifier']}/classify",
                json={"text": data.text}
            )
            drug_task = client.post(
                f"{SERVICES['drug_detector']}/detect",
                json={"text": data.text}
            )

            specialty_resp, drug_resp = await asyncio.gather(specialty_task, drug_task)
            specialty_resp.raise_for_status()
            drug_resp.raise_for_status()

            result = {
                "specialty": specialty_resp.json(),
                "drugs": drug_resp.json(),
            }
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

    # Save to database (shortened input text)
    await database.execute(
        "INSERT INTO predictions (type, input_data, result) VALUES (:t, :i, :r)",
        {
            "t": "prescription",
            "i": json.dumps({"text": data.text[:200]}),
            "r": json.dumps(result)
        }
    )

    return result


@app.get("/history", tags=["History"])
async def get_history(limit: int = 20):
    """Get prediction history"""
    rows = await database.fetch_all(
        "SELECT id, type, result, created_at FROM predictions ORDER BY created_at DESC LIMIT :l",
        {"l": limit}
    )
    return [dict(r) for r in rows]