import requests
import pika
import json
import streamlit as st

from config import RABBITMQ_URL


def http_post(url: str, payload: dict, timeout: int = 15):
    """POST to a microservice. Returns (data, error_str)."""
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Service unavailable"
    except Exception as e:
        return None, str(e)

def http_get(url: str, timeout: int = 10):
    """GET from a microservice. Returns (data, error_str)."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Service unavailable"
    except Exception as e:
        return None, str(e)

def consume_one(queue: str) -> dict | None:
    """
    Pull exactly ONE message from a RabbitMQ queue (basic_get).
    Returns the parsed JSON payload or None if the queue is empty / unreachable.
    """
    try:
        conn    = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = conn.channel()
        channel.queue_declare(queue=queue, durable=True)

        method, _props, body = channel.basic_get(queue=queue, auto_ack=True)
        conn.close()

        if method is None:
            return None
        return json.loads(body)
    except Exception as e:
        st.warning(f"RabbitMQ error: {e}")
        return None

def risk_icon(level: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Very High": "🔴🔴"}.get(level, "⚪")

def stream_controls(key: str):
    """Render ▶️ / ⏹️ buttons. Returns (start_pressed, stop_pressed)."""
    c1, c2, _ = st.columns([1, 1, 6])
    start = c1.button("▶️ Start", key=f"{key}_start", type="primary",
                      disabled=st.session_state[f"run_{key}"])
    stop  = c2.button("⏹️ Stop",  key=f"{key}_stop",
                      disabled=not st.session_state[f"run_{key}"])
    return start, stop

def append_log(key: str, entry: dict, max_len: int = 100):
    st.session_state[key].append(entry)
    if len(st.session_state[key]) > max_len:
        st.session_state[key].pop(0)