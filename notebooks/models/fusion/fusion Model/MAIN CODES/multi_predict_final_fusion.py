
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier, XGBRegressor


# =========================================================
# MODEL BLOCKS
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, input_size, channels=None, kernel_size=3, dropout=0.2, output_size=None):
        super().__init__()
        if channels is None:
            channels = [32, 64]
        if output_size is None:
            output_size = input_size

        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)


class AE(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden1), nn.ReLU(),
            nn.Linear(hidden1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =========================================================
# HELPERS
# =========================================================
def persistence_filter(risk, threshold=0.55, min_len=2):
    out = np.zeros(len(risk), dtype=int)
    count = 0
    for i, r in enumerate(risk):
        if r >= threshold:
            count += 1
        else:
            count = 0
        if count >= min_len:
            out[i] = 1
    return out


def build_residual_feature_df(residuals, residual_score, feature_names):
    df_feat = pd.DataFrame()
    df_feat["residual_mean"] = residuals.mean(axis=1)
    df_feat["residual_max"] = residuals.max(axis=1)
    df_feat["residual_std"] = residuals.std(axis=1)
    df_feat["residual_min"] = residuals.min(axis=1)
    df_feat["residual_range"] = residuals.max(axis=1) - residuals.min(axis=1)
    df_feat["residual_score"] = residual_score

    for i, name in enumerate(feature_names):
        df_feat[f"{name}_residual"] = residuals[:, i]

    sorted_res = np.sort(residuals, axis=1)
    df_feat["top1_residual"] = sorted_res[:, -1]
    df_feat["top2_residual"] = sorted_res[:, -2] if residuals.shape[1] >= 2 else sorted_res[:, -1]
    return df_feat


def rolling_mean_diff(x, short_win=5, long_win=30):
    x = pd.Series(x)
    return (x.rolling(short_win, min_periods=1).mean() - x.rolling(long_win, min_periods=1).mean()).abs().values


def resolve_path(candidates):
    for p in candidates:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


# =========================================================
# DATASET-SPECIFIC PREDICTOR
# =========================================================
class GeneralizedFirePredictor:
    """
    Dataset-specific pipeline:
        input sequence -> TCN -> residual features -> ISO + AE (+ drift for DS3) -> XGBoost risk
    """

    def __init__(self, dataset_type="dataset1", base_dir=None):
        self.dataset_type = dataset_type
        self.device = torch.device("cpu")

        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.alt_dir = Path("/mnt/data")

        if dataset_type == "dataset1":
            self.feature_names = ["Humidity", "Temperature", "MQ139", "TVOC", "eCO2"]
            self.window_size = 60
            self.ae_input_dim = 10
            self.ae_hidden = (32, 16, 8)
            self.xgb_feature_names = ["residual", "risk_tcn", "iso_score", "ae_score"]
            self.tcn_path = resolve_path([
                self.base_dir / "tcn_dataset1_forecaster.pt",
                self.base_dir / "tcn_ssl_forecaster.pt",
                self.alt_dir / "tcn_dataset1_forecaster.pt",
                self.alt_dir / "tcn_ssl_forecaster.pt",
            ])
            self.scaler_path = resolve_path([
                self.base_dir / "scaler_dataset1.pkl",
                self.base_dir / "scaler.pkl",
                self.base_dir / "scaler.pkl1",
                self.alt_dir / "scaler_dataset1.pkl",
                self.alt_dir / "scaler.pkl",
                self.alt_dir / "scaler.pkl1",
            ])
            self.iso_scaler_path = resolve_path([
                self.base_dir / "scaler_iso1.pkl",
                self.alt_dir / "scaler_iso1.pkl",
            ])
            self.iso_path = resolve_path([
                self.base_dir / "iso_dataset1.pkl",
                self.alt_dir / "iso_dataset1.pkl",
            ])
            self.ae_path = resolve_path([
                self.base_dir / "ae_dataset1.pth",
                self.alt_dir / "ae_dataset1.pth",
            ])
            self.xgb_path = resolve_path([
                self.base_dir / "xgb_dataset1.json",
                self.alt_dir / "xgb_dataset1.json",
            ])
            self.use_drift = False

        elif dataset_type == "dataset2":
            self.feature_names = [
                "Temperature_Room",
                "Humidity_Room",
                "CO_Room",
                "CO2_Room",
                "PM25_Room",
                "PM10_Room",
                "VOC_Room_RAW",
            ]
            self.window_size = 60
            self.ae_input_dim = 17
            self.ae_hidden = (64, 32, 16)
            self.xgb_feature_names = [
                "residual_mean", "residual_max", "residual_std", "residual_min", "residual_range", "residual_score",
                "Temperature_Room_residual", "Humidity_Room_residual", "CO_Room_residual", "CO2_Room_residual",
                "PM25_Room_residual", "PM10_Room_residual", "VOC_Room_RAW_residual",
                "top1_residual", "top2_residual", "risk_tcn", "iso_score", "ae_score"
            ]
            self.tcn_path = resolve_path([
                self.base_dir / "tcn_dataset2_forecaster.pt",
                self.alt_dir / "tcn_dataset2_forecaster.pt",
            ])
            self.scaler_path = resolve_path([
                self.base_dir / "scaler_dataset2.pkl",
                self.alt_dir / "scaler_dataset2.pkl",
            ])
            self.iso_scaler_path = resolve_path([
                self.base_dir / "scaler_iso2.pkl",
                self.alt_dir / "scaler_iso2.pkl",
            ])
            self.iso_path = resolve_path([
                self.base_dir / "iso_dataset2.pkl",
                self.alt_dir / "iso_dataset2.pkl",
            ])
            self.ae_path = resolve_path([
                self.base_dir / "ae_dataset2.pth",
                self.alt_dir / "ae_dataset2.pth",
            ])
            self.xgb_path = resolve_path([
                self.base_dir / "xgb_dataset2.json",
                self.alt_dir / "xgb_dataset2.json",
            ])
            self.use_drift = False

        elif dataset_type == "dataset3":
            self.feature_names = ["CO", "CNG", "LPG", "Smoke", "Flame"]
            self.window_size = 60
            self.ae_input_dim = 14
            self.ae_hidden = (32, 16, 8)
            self.xgb_feature_names = [
                "residual_mean", "residual_max", "residual_std", "residual_min", "residual_range", "residual_score",
                "CO_residual", "CNG_residual", "LPG_residual", "Smoke_residual", "Flame_residual",
                "top1_residual", "top2_residual", "risk_tcn", "iso_score", "ae_score", "drift_score", "drift_flag"
            ]
            self.tcn_path = resolve_path([
                self.base_dir / "tcn_dataset3_forecaster.pt",
                self.alt_dir / "tcn_dataset3_forecaster.pt",
            ])
            self.scaler_path = resolve_path([
                self.base_dir / "scaler_dataset3.pkl",
                self.alt_dir / "scaler_dataset3.pkl",
            ])
            self.iso_scaler_path = resolve_path([
                self.base_dir / "scaler_iso3.pkl",
                self.alt_dir / "scaler_iso3.pkl",
            ])
            self.iso_path = resolve_path([
                self.base_dir / "iso_dataset3.pkl",
                self.alt_dir / "iso_dataset3.pkl",
            ])
            self.ae_path = resolve_path([
                self.base_dir / "ae_dataset3.pth",
                self.alt_dir / "ae_dataset3.pth",
            ])
            self.xgb_path = resolve_path([
                self.base_dir / "xgb_dataset3.json",
                self.alt_dir / "xgb_dataset3.json",
            ])
            self.use_drift = True
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        self.input_size = len(self.feature_names)
        self.scaler = self._load_pickle(self.scaler_path)
        self.iso_scaler = self._load_pickle(self.iso_scaler_path)
        self.iso = self._load_pickle(self.iso_path)
        self.tcn = self._load_tcn()
        self.ae = self._load_ae()
        self.xgb = self._load_xgb()

    def _load_pickle(self, path):
        if path is None:
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_tcn(self):
        model = TCNForecaster(
            input_size=self.input_size,
            channels=[32, 64],
            kernel_size=3,
            dropout=0.2,
            output_size=self.input_size,
        ).to(self.device)

        if self.tcn_path is not None:
            state_dict = torch.load(self.tcn_path, map_location=self.device)
            model.load_state_dict(state_dict)

        model.eval()
        return model

    def _load_ae(self):
        h1, h2, z = self.ae_hidden
        model = AE(self.ae_input_dim, h1, h2, z).to(self.device)

        if self.ae_path is not None:
            state_dict = torch.load(self.ae_path, map_location=self.device)
            model.load_state_dict(state_dict)

        model.eval()
        return model

    def _load_xgb(self):
        if self.xgb_path is None:
            return None

        model = XGBClassifier()
        model.load_model(str(self.xgb_path))
        return model

    def _prepare_df(self, df):
        out = df.copy()
        for col in self.feature_names:
            if col not in out.columns:
                out[col] = 0.0
        out = out[self.feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return out

    def _scale_features(self, values):
        if self.scaler is None:
            return values
        return self.scaler.transform(values)

    def _tcn_windows(self, X_scaled):
        if len(X_scaled) < self.window_size:
            return None, None, None

        risks_tcn = []
        residual_scores = []
        residual_rows = []

        for end in range(self.window_size, len(X_scaled) + 1):
            window = X_scaled[end - self.window_size:end]
            x = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                pred_scaled = self.tcn(x).cpu().numpy()[0]

            target_scaled = window[-1]
            residual = np.abs(pred_scaled - target_scaled)

            residual_rows.append(residual)
            residual_score = float(np.mean(residual))
            residual_scores.append(residual_score)

            # Simple normalized TCN anomaly proxy
            risks_tcn.append(float(np.clip(residual_score * 5.0, 0.0, 1.0)))

        return np.array(risks_tcn), np.array(residual_scores), np.vstack(residual_rows)

    def _feature_frame(self, residuals, residual_score, risk_tcn):
        if self.dataset_type == "dataset1":
            return pd.DataFrame({
                "residual": residual_score,
                "risk_tcn": risk_tcn,
            })

        feat = build_residual_feature_df(residuals, residual_score, self.feature_names)
        feat["risk_tcn"] = risk_tcn
        return feat

    def _add_iso_score(self, feat_df):
        if self.iso is None or self.iso_scaler is None:
            feat_df["iso_score"] = 0.0
            return feat_df

        base_cols = [c for c in feat_df.columns if c not in ("iso_score", "ae_score", "drift_score", "drift_flag")]
        X_iso = feat_df[base_cols].copy()
        X_iso_scaled = self.iso_scaler.transform(X_iso)
        feat_df["iso_score"] = -self.iso.decision_function(X_iso_scaled)  # higher = more anomalous
        return feat_df

    def _add_ae_score(self, feat_df):
        base_cols = [c for c in feat_df.columns if c not in ("iso_score", "ae_score", "drift_score", "drift_flag")]
        X = feat_df[base_cols].copy()

        if X.shape[1] < self.ae_input_dim:
            for i in range(self.ae_input_dim - X.shape[1]):
                X[f"_pad_{i}"] = 0.0
        elif X.shape[1] > self.ae_input_dim:
            X = X.iloc[:, :self.ae_input_dim]

        x_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self.ae(x_tensor)
            err = torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()

        feat_df["ae_score"] = err
        return feat_df

    def _add_drift_features(self, feat_df):
        if not self.use_drift:
            return feat_df

        feat_df["drift_score"] = rolling_mean_diff(feat_df["risk_tcn"].values, short_win=5, long_win=30)
        feat_df["drift_flag"] = (feat_df["drift_score"] > 0.04).astype(int)
        return feat_df

    def _run_xgb(self, feat_df):
        if self.xgb is None:
            # fallback if fusion model file is missing
            if "risk_tcn" in feat_df.columns:
                return np.clip(feat_df["risk_tcn"].values, 0.0, 1.0)
            if "residual_score" in feat_df.columns:
                return np.clip(feat_df["residual_score"].values, 0.0, 1.0)
            return np.zeros(len(feat_df))

        use_cols = []
        for col in self.xgb_feature_names:
            if col in feat_df.columns:
                use_cols.append(col)
            else:
                feat_df[col] = 0.0
                use_cols.append(col)

        X = feat_df[use_cols].copy()
        try:
            proba = self.xgb.predict_proba(X)[:, 1]
            return np.clip(proba, 0.0, 1.0)
        except Exception:
            pred = self.xgb.predict(X)
            return np.clip(pred.astype(float), 0.0, 1.0)

    def predict(self, df) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
            risks_full: array length len(df)
            alarms_full: array length len(df)
            extra: last-step diagnostics
        """
        df = self._prepare_df(df)

        if len(df) < self.window_size:
            n = len(df)
            return np.zeros(n), np.zeros(n, dtype=int), {
                "residual_score": 0.0,
                "iso_score": 0.0,
                "ae_score": 0.0,
                "drift_score": 0.0,
            }

        X_values = df.values
        X_scaled = self._scale_features(X_values)

        risk_tcn, residual_score, residuals = self._tcn_windows(X_scaled)
        feat_df = self._feature_frame(residuals, residual_score, risk_tcn)
        feat_df = self._add_iso_score(feat_df)
        feat_df = self._add_ae_score(feat_df)
        feat_df = self._add_drift_features(feat_df)

        final_risk = self._run_xgb(feat_df)
        alarms = persistence_filter(final_risk, threshold=0.55, min_len=2)

        # pad front so output length matches input df length
        pad_len = len(df) - len(final_risk)
        risks_full = np.concatenate([np.zeros(pad_len), final_risk])
        alarms_full = np.concatenate([np.zeros(pad_len, dtype=int), alarms])

        last_idx = len(feat_df) - 1
        extra = {
            "residual_score": float(feat_df["residual_score"].iloc[last_idx]) if "residual_score" in feat_df.columns else 0.0,
            "iso_score": float(feat_df["iso_score"].iloc[last_idx]) if "iso_score" in feat_df.columns else 0.0,
            "ae_score": float(feat_df["ae_score"].iloc[last_idx]) if "ae_score" in feat_df.columns else 0.0,
            "drift_score": float(feat_df["drift_score"].iloc[last_idx]) if "drift_score" in feat_df.columns else 0.0,
        }
        return risks_full, alarms_full, extra


# =========================================================
# GLOBAL MULTI-MODEL FUSION
# =========================================================
class FusionRiskCombiner:
    """
    True multi-model fusion layer on top of dataset-specific pipelines.
    Uses:
      1) learned fusion model if available
      2) otherwise weighted fusion fallback
    """

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.alt_dir = Path("/mnt/data")

        self.fusion_model_path = resolve_path([
            self.base_dir / "global_fusion_xgb.json",
            self.base_dir / "fusion_xgb.json",
            self.alt_dir / "global_fusion_xgb.json",
            self.alt_dir / "fusion_xgb.json",
        ])
        self.model: Optional[XGBRegressor] = None
        self.model_type = "weighted_fallback"

        if self.fusion_model_path is not None:
            try:
                self.model = XGBRegressor()
                self.model.load_model(str(self.fusion_model_path))
                self.model_type = "learned_xgb"
            except Exception:
                self.model = None
                self.model_type = "weighted_fallback"

        self.default_weights = {
            "ds1": 0.33,
            "ds2": 0.34,
            "ds3": 0.33,
        }

    def predict(self, ds1_risk, ds2_risk, ds3_risk, ds1_ready=True, ds2_ready=True, ds3_ready=True):
        values = {
            "ds1": float(ds1_risk),
            "ds2": float(ds2_risk),
            "ds3": float(ds3_risk),
        }
        ready = {
            "ds1": bool(ds1_ready),
            "ds2": bool(ds2_ready),
            "ds3": bool(ds3_ready),
        }

        if self.model is not None:
            feat = np.array([[values["ds1"], values["ds2"], values["ds3"]]], dtype=float)
            pred = float(self.model.predict(feat)[0])
            return float(np.clip(pred, 0.0, 1.0))

        # weighted fallback over only ready pipelines
        active_weights = []
        active_values = []
        for key in ("ds1", "ds2", "ds3"):
            if ready[key]:
                active_weights.append(self.default_weights[key])
                active_values.append(values[key])

        if not active_values:
            return 0.0

        w = np.array(active_weights, dtype=float)
        x = np.array(active_values, dtype=float)
        w = w / w.sum()
        return float(np.clip(np.sum(w * x), 0.0, 1.0))
