
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from multi_predict_final_fusion import GeneralizedFirePredictor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

DATASET1_FILES = [
    BASE_DIR / "datasets" / "carton_1.csv",
    BASE_DIR / "datasets" / "carton_2.csv",
    BASE_DIR / "datasets" / "clothing_1.csv",
    BASE_DIR / "datasets" / "clothing_2.csv",
    BASE_DIR / "datasets" / "electrical_1.csv",
    BASE_DIR / "datasets" / "electrical_2.csv",
    BASE_DIR / "datasets" / "electrical_3.csv",
    BASE_DIR / "datasets" / "electrical_4.csv",
]

DATASET2_PATH = BASE_DIR / "datasets" / "Indoor Fire Dataset with Distributed Multi-Sensor Nodes_Industrial Hall.csv"

DATASET3_FILES = {
    "CNG": BASE_DIR / "datasets" / "cng_sensor.xlsx",
    "CO": BASE_DIR / "datasets" / "co_sensor.xlsx",
    "Flame": BASE_DIR / "datasets" / "flame_sensor.xlsx",
    "LPG": BASE_DIR / "datasets" / "lpg_sensor.xlsx",
    "Smoke": BASE_DIR / "datasets" / "smoke_sensor.xlsx",
}

OUT_FUSION_DATASET = BASE_DIR / "fusion_training_dataset.csv"
OUT_FUSION_MODEL = BASE_DIR / "global_fusion_xgb.json"
OUT_METRICS = BASE_DIR / "fusion_metrics.txt"

CANDIDATE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().replace("%", "").replace("#", "").replace("  ", " ").strip()
        for c in df.columns
    ]
    return df


def ensure_files_exist(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_dataset1() -> pd.DataFrame:
    ensure_files_exist(DATASET1_FILES)
    frames = []

    for fp in DATASET1_FILES:
        df = pd.read_csv(fp)
        df = clean_columns(df)

        rename_map = {}
        for col in df.columns:
            c = col.replace(" ", "")
            if c.lower() == "humidity":
                rename_map[col] = "Humidity"
            elif c.lower() == "temperature":
                rename_map[col] = "Temperature"
            elif c.lower() == "mq139":
                rename_map[col] = "MQ139"
            elif c.lower() == "tvoc":
                rename_map[col] = "TVOC"
            elif c.lower() == "eco2":
                rename_map[col] = "eCO2"
            elif c.lower() == "status":
                rename_map[col] = "Status"
            elif c.lower() == "time":
                rename_map[col] = "Time"

        df = df.rename(columns=rename_map)
        needed = ["Humidity", "Temperature", "MQ139", "TVOC", "eCO2", "Status"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"Dataset1 file {fp.name} is missing required column: {col}")

        df = safe_numeric(df, ["Humidity", "Temperature", "MQ139", "TVOC", "eCO2", "Status"])
        df = df.dropna(subset=["Humidity", "Temperature", "MQ139", "TVOC", "eCO2"]).copy()
        df["source"] = "dataset1"
        df["scenario_file"] = fp.name
        df["event_label"] = (df["Status"].fillna(0) > 0).astype(int)

        frames.append(df[["Humidity", "Temperature", "MQ139", "TVOC", "eCO2", "event_label", "source", "scenario_file"]])

    return pd.concat(frames, ignore_index=True)


def load_dataset2() -> pd.DataFrame:
    ensure_files_exist([DATASET2_PATH])
    df = pd.read_csv(DATASET2_PATH)
    df = clean_columns(df)

    required = [
        "Date", "Temperature_Room", "Humidity_Room", "CO_Room", "CO2_Room",
        "PM25_Room", "PM10_Room", "VOC_Room_RAW", "fire"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dataset2 is missing required column: {col}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).copy()

    num_cols = ["Temperature_Room", "Humidity_Room", "CO_Room", "CO2_Room", "PM25_Room", "PM10_Room", "VOC_Room_RAW", "fire"]
    df = safe_numeric(df, num_cols)
    df = df.dropna(subset=["Temperature_Room", "Humidity_Room", "CO_Room", "CO2_Room", "PM25_Room", "PM10_Room", "VOC_Room_RAW"]).copy()

    if "Valid_Experiment" in df.columns:
        df["Valid_Experiment"] = pd.to_numeric(df["Valid_Experiment"], errors="coerce").fillna(1)
        df = df[df["Valid_Experiment"] == 1].copy()

    agg = (
        df.groupby("Date", as_index=False)
          .agg({
              "Temperature_Room": "mean",
              "Humidity_Room": "mean",
              "CO_Room": "mean",
              "CO2_Room": "mean",
              "PM25_Room": "mean",
              "PM10_Room": "mean",
              "VOC_Room_RAW": "mean",
              "fire": "max",
          })
          .sort_values("Date")
          .reset_index(drop=True)
    )

    agg["source"] = "dataset2"
    agg["event_label"] = (agg["fire"].fillna(0) > 0).astype(int)

    return agg[[
        "Temperature_Room", "Humidity_Room", "CO_Room", "CO2_Room",
        "PM25_Room", "PM10_Room", "VOC_Room_RAW", "event_label", "source"
    ]]


def read_sensor_excel(path: Path, target_name: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = clean_columns(df)

    required = ["time", "data_value"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing required column: {col}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df.dropna(subset=["time", "data_value"]).copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "data_value"]].rename(columns={"data_value": target_name})


def load_dataset3() -> pd.DataFrame:
    ensure_files_exist(list(DATASET3_FILES.values()))

    cng = read_sensor_excel(DATASET3_FILES["CNG"], "CNG")
    co = read_sensor_excel(DATASET3_FILES["CO"], "CO")
    flame = read_sensor_excel(DATASET3_FILES["Flame"], "Flame")
    lpg = read_sensor_excel(DATASET3_FILES["LPG"], "LPG")
    smoke = read_sensor_excel(DATASET3_FILES["Smoke"], "Smoke")

    base = flame.sort_values("time").copy()
    for other in [co, cng, lpg, smoke]:
        base = pd.merge_asof(
            base.sort_values("time"),
            other.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("30s")
        )

    base = base.dropna().reset_index(drop=True)

    q = base[["CO", "CNG", "LPG", "Smoke", "Flame"]].quantile([0.90, 0.95])
    th_co = float(q.loc[0.95, "CO"])
    th_cng = float(q.loc[0.95, "CNG"])
    th_lpg = float(q.loc[0.95, "LPG"])
    th_smoke = float(q.loc[0.95, "Smoke"])
    th_flame = float(q.loc[0.90, "Flame"])

    trigger_count = (
        (base["CO"] >= th_co).astype(int) +
        (base["CNG"] >= th_cng).astype(int) +
        (base["LPG"] >= th_lpg).astype(int) +
        (base["Smoke"] >= th_smoke).astype(int)
    )

    base["event_label"] = (((base["Flame"] >= th_flame) & (trigger_count >= 1)) | (trigger_count >= 2)).astype(int)
    base["source"] = "dataset3"

    return base[["CO", "CNG", "LPG", "Smoke", "Flame", "event_label", "source"]]


def to_canonical_from_dataset1(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["humidity_rh"] = df["Humidity"]
    out["temperature_c"] = df["Temperature"]
    out["mq2"] = df["MQ139"]
    out["mq4"] = 0.0
    out["mq6"] = 0.0
    out["mq9"] = 0.0
    out["pm25_ugm3"] = 0.0
    out["pm10_ugm3"] = 0.0
    out["tvoc_ppb"] = df["TVOC"]
    out["eco2_ppm"] = df["eCO2"]
    out["uv"] = 0.0
    out["label"] = df["event_label"].astype(int)
    out["source"] = df["source"]
    return out


def to_canonical_from_dataset2(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["humidity_rh"] = df["Humidity_Room"]
    out["temperature_c"] = df["Temperature_Room"]
    out["mq2"] = 0.0
    out["mq4"] = 0.0
    out["mq6"] = 0.0
    out["mq9"] = df["CO_Room"]
    out["pm25_ugm3"] = df["PM25_Room"]
    out["pm10_ugm3"] = df["PM10_Room"]
    out["tvoc_ppb"] = df["VOC_Room_RAW"]
    out["eco2_ppm"] = df["CO2_Room"]
    out["uv"] = 0.0
    out["label"] = df["event_label"].astype(int)
    out["source"] = df["source"]
    return out


def to_canonical_from_dataset3(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["humidity_rh"] = 0.0
    out["temperature_c"] = 0.0
    out["mq2"] = df["Smoke"]
    out["mq4"] = df["CNG"]
    out["mq6"] = df["LPG"]
    out["mq9"] = df["CO"]
    out["pm25_ugm3"] = 0.0
    out["pm10_ugm3"] = 0.0
    out["tvoc_ppb"] = 0.0
    out["eco2_ppm"] = 0.0
    out["uv"] = df["Flame"]
    out["label"] = df["event_label"].astype(int)
    out["source"] = df["source"]
    return out


def map_for_predictors(canon: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds1 = pd.DataFrame({
        "Humidity": canon["humidity_rh"],
        "Temperature": canon["temperature_c"],
        "MQ139": canon["mq2"],
        "TVOC": canon["tvoc_ppb"],
        "eCO2": canon["eco2_ppm"].replace(0, 400),
    })

    ds2 = pd.DataFrame({
        "Temperature_Room": canon["temperature_c"],
        "Humidity_Room": canon["humidity_rh"],
        "CO_Room": canon["mq9"],
        "CO2_Room": canon["eco2_ppm"],
        "PM25_Room": canon["pm25_ugm3"],
        "PM10_Room": canon["pm10_ugm3"],
        "VOC_Room_RAW": canon["tvoc_ppb"],
    })

    ds3 = pd.DataFrame({
        "CO": canon["mq9"],
        "CNG": canon["mq4"],
        "LPG": canon["mq6"],
        "Smoke": canon["mq2"],
        "Flame": canon["uv"],
    })

    return ds1, ds2, ds3


def make_fusion_rows_for_source(
    canon_df: pd.DataFrame,
    predictor_ds1: GeneralizedFirePredictor,
    predictor_ds2: GeneralizedFirePredictor,
    predictor_ds3: GeneralizedFirePredictor
) -> pd.DataFrame:
    ds1_df, ds2_df, ds3_df = map_for_predictors(canon_df)

    r1, _, _ = predictor_ds1.predict(ds1_df)
    r2, _, _ = predictor_ds2.predict(ds2_df)
    r3, _, _ = predictor_ds3.predict(ds3_df)

    out = pd.DataFrame({
        "ds1_risk": r1,
        "ds2_risk": r2,
        "ds3_risk": r3,
        "label": canon_df["label"].astype(int).values,
        "source": canon_df["source"].values,
    })

    active = (out[["ds1_risk", "ds2_risk", "ds3_risk"]].sum(axis=1) > 0)
    return out[active].reset_index(drop=True)


def chronological_split(df: pd.DataFrame, frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = max(1, int(n * frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def pick_best_threshold(y_true: pd.Series, proba: np.ndarray) -> Tuple[float, dict, np.ndarray]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_pred = None
    best_metrics = None

    for th in CANDIDATE_THRESHOLDS:
        pred = (proba >= th).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1_score": f1_score(y_true, pred, zero_division=0),
        }
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = th
            best_pred = pred
            best_metrics = metrics

    return best_threshold, best_metrics, best_pred


def train_fusion_model(fusion_df: pd.DataFrame) -> None:
    train_parts = []
    test_parts = []

    for src in fusion_df["source"].unique():
        sub = fusion_df[fusion_df["source"] == src].reset_index(drop=True)
        tr, te = chronological_split(sub, frac=0.8)
        train_parts.append(tr)
        test_parts.append(te)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train = train_df[["ds1_risk", "ds2_risk", "ds3_risk"]]
    y_train = train_df["label"].astype(int)

    X_test = test_df[["ds1_risk", "ds2_risk", "ds3_risk"]]
    y_test = test_df["label"].astype(int)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    print("\nTrain label counts:")
    print(y_train.value_counts(dropna=False).sort_index())

    print("\nTest label counts:")
    print(y_test.value_counts(dropna=False).sort_index())

    print(f"\nscale_pos_weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)
    model.save_model(str(OUT_FUSION_MODEL))

    proba = model.predict_proba(X_test)[:, 1]
    best_threshold, best_metrics, pred = pick_best_threshold(y_test, proba)

    print("\nPredicted label counts:")
    print(pd.Series(pred).value_counts(dropna=False).sort_index())

    cm = confusion_matrix(y_test, pred)

    metrics = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_positive": pos,
        "train_negative": neg,
        "scale_pos_weight": float(scale_pos_weight),
        "best_threshold": float(best_threshold),
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
    }

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("GLOBAL FUSION MODEL METRICS\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTrain label counts:\n")
        f.write(str(y_train.value_counts(dropna=False).sort_index()))
        f.write("\n\nTest label counts:\n")
        f.write(str(y_test.value_counts(dropna=False).sort_index()))
        f.write("\n\nPredicted label counts:\n")
        f.write(str(pd.Series(pred).value_counts(dropna=False).sort_index()))
        f.write("\n\nConfusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(classification_report(y_test, pred, zero_division=0))

    print("\nConfusion matrix:")
    print(cm)

    print("\nSaved:")
    print(f" - Fusion dataset: {OUT_FUSION_DATASET}")
    print(f" - Fusion model:   {OUT_FUSION_MODEL}")
    print(f" - Metrics:        {OUT_METRICS}")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


def main():
    print("Loading trained dataset-specific predictors...")
    predictor_ds1 = GeneralizedFirePredictor(dataset_type="dataset1", base_dir=BASE_DIR)
    predictor_ds2 = GeneralizedFirePredictor(dataset_type="dataset2", base_dir=BASE_DIR)
    predictor_ds3 = GeneralizedFirePredictor(dataset_type="dataset3", base_dir=BASE_DIR)

    print("Loading Dataset 1...")
    d1 = load_dataset1()
    c1 = to_canonical_from_dataset1(d1)
    f1 = make_fusion_rows_for_source(c1, predictor_ds1, predictor_ds2, predictor_ds3)
    print(f"Dataset 1 fusion rows: {len(f1)}")

    print("Loading Dataset 2...")
    d2 = load_dataset2()
    c2 = to_canonical_from_dataset2(d2)
    f2 = make_fusion_rows_for_source(c2, predictor_ds1, predictor_ds2, predictor_ds3)
    print(f"Dataset 2 fusion rows: {len(f2)}")

    print("Loading Dataset 3...")
    d3 = load_dataset3()
    c3 = to_canonical_from_dataset3(d3)
    f3 = make_fusion_rows_for_source(c3, predictor_ds1, predictor_ds2, predictor_ds3)
    print(f"Dataset 3 fusion rows: {len(f3)}")

    fusion_df = pd.concat([f1, f2, f3], ignore_index=True)
    fusion_df.to_csv(OUT_FUSION_DATASET, index=False)
    print(f"Total fusion rows: {len(fusion_df)}")

    train_fusion_model(fusion_df)
    print("\nDone. Your app will now use global_fusion_xgb.json automatically.")


if __name__ == "__main__":
    main()
