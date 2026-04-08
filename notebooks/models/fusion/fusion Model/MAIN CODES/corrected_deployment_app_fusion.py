
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, send_file

import firebase_admin
from firebase_admin import credentials, db

# =========================================================
# IMPORT PREDICTORS AND FUSION COMBINER
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

for p in [CURRENT_DIR, PARENT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from multi_predict_final_fusion import GeneralizedFirePredictor, FusionRiskCombiner
except Exception:
    from multi_predict_final import GeneralizedFirePredictor, FusionRiskCombiner


# =========================================================
# CONFIG
# =========================================================
FIREBASE_DB_URL = "https://firesens-1a708-default-rtdb.firebaseio.com/"
SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"

SENSOR_PATH = "/fireSensors/current"
RISK_PATH = "/fireRisk/current"

POLL_INTERVAL_SEC = 2
HISTORY_MAXLEN = 300

# =========================================================
# FLASK
# =========================================================
app = Flask(__name__)

# =========================================================
# GLOBAL MODELS / BUFFERS / STATE
# =========================================================
predictor_ds1: Optional[GeneralizedFirePredictor] = None
predictor_ds2: Optional[GeneralizedFirePredictor] = None
predictor_ds3: Optional[GeneralizedFirePredictor] = None
fusion_combiner: Optional[FusionRiskCombiner] = None

buffers: Dict[str, deque] = {}
risk_history = deque(maxlen=HISTORY_MAXLEN)

latest_state: Dict[str, Any] = {
    "status": "starting",
    "message": "System booting",
    "window_fill": {"ds1": 0, "ds2": 0, "ds3": 0},
    "window_required": {"ds1": 60, "ds2": 60, "ds3": 60},
    "ds1": {"risk_score": 0.0, "risk_percent": 0.0, "alarm": 0, "risk_level": "LOW", "ready": False},
    "ds2": {"risk_score": 0.0, "risk_percent": 0.0, "alarm": 0, "risk_level": "LOW", "ready": False},
    "ds3": {"risk_score": 0.0, "risk_percent": 0.0, "alarm": 0, "risk_level": "LOW", "ready": False},
    "final_fused": {"risk_score": 0.0, "risk_percent": 0.0, "alarm": 0, "risk_level": "LOW", "ready": False},
    "raw_sensor": {},
    "mapped_ds1": {},
    "mapped_ds2": {},
    "mapped_ds3": {},
    "all_sensor_fields": {},
    "model_notes": {},
    "last_update": None,
}


# =========================================================
# HELPERS
# =========================================================
def init_firebase() -> None:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


def load_artifacts() -> None:
    global predictor_ds1, predictor_ds2, predictor_ds3, fusion_combiner, buffers, latest_state

    predictor_ds1 = GeneralizedFirePredictor(dataset_type="dataset1")
    predictor_ds2 = GeneralizedFirePredictor(dataset_type="dataset2")
    predictor_ds3 = GeneralizedFirePredictor(dataset_type="dataset3")
    fusion_combiner = FusionRiskCombiner()

    buffers = {
        "ds1": deque(maxlen=int(predictor_ds1.window_size)),
        "ds2": deque(maxlen=int(predictor_ds2.window_size)),
        "ds3": deque(maxlen=int(predictor_ds3.window_size)),
    }

    latest_state["window_required"] = {
        "ds1": int(predictor_ds1.window_size),
        "ds2": int(predictor_ds2.window_size),
        "ds3": int(predictor_ds3.window_size),
    }
    latest_state["model_notes"] = {
        "ds1_window": f"dataset1 window size = {predictor_ds1.window_size}",
        "ds2_window": f"dataset2 window size = {predictor_ds2.window_size}",
        "ds3_window": f"dataset3 window size = {predictor_ds3.window_size}",
        "fusion_mode": getattr(fusion_combiner, "model_type", "weighted_fallback"),
        "system_type": "true multi-model fusion system",
        "fusion_description": "DS1 risk + DS2 risk + DS3 risk -> fused final risk",
        "ds1_iso": "active when iso_dataset1.pkl exists",
        "ds2_iso": "active when iso_dataset2.pkl exists",
        "ds3_iso": "active when iso_dataset3.pkl exists",
    }

    print("Predictors and fusion combiner loaded successfully.")
    print(
        f"Window sizes -> ds1={predictor_ds1.window_size}, "
        f"ds2={predictor_ds2.window_size}, ds3={predictor_ds3.window_size}"
    )
    print(f"Fusion mode -> {latest_state['model_notes']['fusion_mode']}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def classify_risk_level(risk: float) -> str:
    if risk >= 0.75:
        return "HIGH"
    if risk >= 0.45:
        return "MEDIUM"
    return "LOW"


def fetch_sensor_snapshot() -> Optional[Dict[str, Any]]:
    ref = db.reference(SENSOR_PATH)
    data = ref.get()
    return data if isinstance(data, dict) else None


def write_risk_to_firebase(payload: Dict[str, Any]) -> None:
    ref = db.reference(RISK_PATH)
    ref.set(payload)


def normalize_sensor_json(sensor_json: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for k, v in sensor_json.items():
        if isinstance(v, (int, float)):
            normalized[k] = float(v)
        else:
            try:
                normalized[k] = float(v)
            except Exception:
                normalized[k] = v
    return normalized


def map_rows(sensor_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    normalized = normalize_sensor_json(sensor_json)

    raw = {
        "Humidity": safe_float(normalized.get("humidity_rh", normalized.get("Humidity", 0.0))),
        "Temp": safe_float(normalized.get("temperature_c", normalized.get("Temp", 0.0))),
        "MQ2": safe_float(normalized.get("mq2", normalized.get("MQ2", 0.0))),
        "MQ4": safe_float(normalized.get("mq4", normalized.get("MQ4", 0.0))),
        "MQ6": safe_float(normalized.get("mq6", normalized.get("MQ6", 0.0))),
        "MQ9": safe_float(normalized.get("mq9", normalized.get("MQ9", 0.0))),
        "PM2.5": safe_float(normalized.get("pm25_ugm3", normalized.get("PM2.5", 0.0))),
        "PM10": safe_float(normalized.get("pm10_ugm3", normalized.get("PM10", 0.0))),
        "TVOC": safe_float(normalized.get("tvoc_ppb", normalized.get("TVOC", 0.0))),
        "eCO2": safe_float(normalized.get("eco2_ppm", normalized.get("eCO2", 0.0))),
        "UV": safe_float(normalized.get("uv", normalized.get("UV", 0.0))),
    }

    row_ds1 = {
        "Humidity": raw["Humidity"],
        "Temperature": raw["Temp"],
        "MQ139": raw["MQ2"],
        "TVOC": raw["TVOC"],
        "eCO2": raw["eCO2"] if raw["eCO2"] > 0 else 400.0,
    }

    row_ds2 = {
        "Temperature_Room": raw["Temp"],
        "Humidity_Room": raw["Humidity"],
        "CO_Room": raw["MQ9"],
        "CO2_Room": raw["eCO2"] if raw["eCO2"] > 0 else raw["TVOC"],
        "PM25_Room": raw["PM2.5"],
        "PM10_Room": raw["PM10"] if raw["PM10"] > 0 else raw["PM2.5"] * 2.0,
        "VOC_Room_RAW": raw["TVOC"],
    }

    row_ds3 = {
        "CO": raw["MQ9"],
        "CNG": raw["MQ4"],
        "LPG": raw["MQ6"],
        "Smoke": raw["MQ2"],
        "Flame": raw["UV"],
    }

    return normalized, raw, row_ds1, row_ds2, row_ds3


def predict_one(name: str, predictor: GeneralizedFirePredictor, buffer: deque) -> Dict[str, Any]:
    if len(buffer) < predictor.window_size:
        return {
            "risk_score": 0.0,
            "risk_percent": 0.0,
            "alarm": 0,
            "risk_level": "LOW",
            "ready": False,
            "fill": len(buffer),
            "required": int(predictor.window_size),
            "residual_score": 0.0,
            "iso_score": 0.0,
            "ae_score": 0.0,
            "drift_score": 0.0,
        }

    df = pd.DataFrame(list(buffer))
    risks, alarms, extra = predictor.predict(df)

    risk_value = float(risks[-1]) if len(risks) > 0 else 0.0
    alarm_value = int(alarms[-1]) if alarms is not None and len(alarms) > 0 else 0

    out = {
        "risk_score": risk_value,
        "risk_percent": round(risk_value * 100.0, 2),
        "alarm": alarm_value,
        "risk_level": classify_risk_level(risk_value),
        "ready": True,
        "fill": len(buffer),
        "required": int(predictor.window_size),
        "residual_score": 0.0,
        "iso_score": 0.0,
        "ae_score": 0.0,
        "drift_score": 0.0,
    }

    if isinstance(extra, dict):
        for k in ("residual_score", "iso_score", "ae_score", "drift_score"):
            if k in extra:
                try:
                    out[k] = float(extra[k])
                except Exception:
                    out[k] = 0.0
    return out


def build_final_fused(ds1_out: Dict[str, Any], ds2_out: Dict[str, Any], ds3_out: Dict[str, Any]) -> Dict[str, Any]:
    final_risk = fusion_combiner.predict(
        ds1_out["risk_score"],
        ds2_out["risk_score"],
        ds3_out["risk_score"],
        ds1_out["ready"],
        ds2_out["ready"],
        ds3_out["ready"],
    )

    final_alarm = 1 if final_risk >= 0.55 else 0
    any_ready = ds1_out["ready"] or ds2_out["ready"] or ds3_out["ready"]

    return {
        "risk_score": final_risk,
        "risk_percent": round(final_risk * 100.0, 2),
        "alarm": final_alarm,
        "risk_level": classify_risk_level(final_risk),
        "ready": any_ready,
    }


def inference_step(sensor_json: Dict[str, Any]) -> None:
    global latest_state

    normalized, raw, row_ds1, row_ds2, row_ds3 = map_rows(sensor_json)

    buffers["ds1"].append(row_ds1)
    buffers["ds2"].append(row_ds2)
    buffers["ds3"].append(row_ds3)

    ds1_out = predict_one("ds1", predictor_ds1, buffers["ds1"])
    ds2_out = predict_one("ds2", predictor_ds2, buffers["ds2"])
    ds3_out = predict_one("ds3", predictor_ds3, buffers["ds3"])
    final_fused = build_final_fused(ds1_out, ds2_out, ds3_out)

    ready_all = ds1_out["ready"] and ds2_out["ready"] and ds3_out["ready"]
    any_ready = ds1_out["ready"] or ds2_out["ready"] or ds3_out["ready"]

    if ready_all:
        status = "running"
        message = "All pipelines ready, fused system active"
    elif any_ready:
        status = "partial_ready"
        message = "Some pipelines ready, fused system using available model outputs"
    else:
        status = "warming_up"
        message = "Collecting sequence data for one or more pipelines"

    latest_state = {
        "status": status,
        "message": message,
        "window_fill": {
            "ds1": len(buffers["ds1"]),
            "ds2": len(buffers["ds2"]),
            "ds3": len(buffers["ds3"]),
        },
        "window_required": {
            "ds1": int(predictor_ds1.window_size),
            "ds2": int(predictor_ds2.window_size),
            "ds3": int(predictor_ds3.window_size),
        },
        "ds1": ds1_out,
        "ds2": ds2_out,
        "ds3": ds3_out,
        "final_fused": final_fused,
        "raw_sensor": raw,
        "all_sensor_fields": normalized,
        "mapped_ds1": row_ds1,
        "mapped_ds2": row_ds2,
        "mapped_ds3": row_ds3,
        "model_notes": latest_state.get("model_notes", {}),
        "last_update": int(time.time()),
    }

    risk_history.append({
        "ts": latest_state["last_update"],
        "ds1_risk": ds1_out["risk_percent"],
        "ds2_risk": ds2_out["risk_percent"],
        "ds3_risk": ds3_out["risk_percent"],
        "final_risk": final_fused["risk_percent"],
    })

    write_risk_to_firebase(latest_state)


def background_worker() -> None:
    global latest_state
    while True:
        try:
            sensor_json = fetch_sensor_snapshot()
            if sensor_json is None:
                latest_state["status"] = "waiting_for_sensor"
                latest_state["message"] = "No sensor data found in Firebase"
                latest_state["last_update"] = int(time.time())
            else:
                inference_step(sensor_json)
        except Exception as e:
            latest_state["status"] = "error"
            latest_state["message"] = str(e)
            latest_state["last_update"] = int(time.time())
            print(f"[ERROR] {e}", flush=True)

        time.sleep(POLL_INTERVAL_SEC)


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def index():
    html_candidates = [
        CURRENT_DIR / "index_final_fusion.html",
        CURRENT_DIR / "index_final.html",
        CURRENT_DIR / "index.html",
        CURRENT_DIR / "templates" / "index.html",
    ]
    for path in html_candidates:
        if path.exists():
            return send_file(path)
    return jsonify({
        "message": "Dashboard file not found. Use /api/latest or place index_final_fusion.html next to this script."
    }), 200


@app.route("/api/latest")
def api_latest():
    return jsonify(latest_state)


@app.route("/api/history")
def api_history():
    return jsonify(list(risk_history))


@app.route("/health")
def health():
    return jsonify({
        "status": latest_state.get("status"),
        "last_update": latest_state.get("last_update"),
        "windows": latest_state.get("window_fill"),
        "final_fused": latest_state.get("final_fused"),
    })


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Starting fused fire-risk deployment app...")
    init_firebase()
    load_artifacts()

    worker = threading.Thread(target=background_worker, daemon=True)
    worker.start()

    app.run(host="0.0.0.0", port=5000, debug=False)
