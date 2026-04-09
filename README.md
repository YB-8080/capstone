# Capstone Project: Intelligent IoT Multi-Sensor Fire Risk Detection System

## Overview
This repository houses the machine learning architecture and real-time deployment pipeline for a robust, multi-sensor fire risk detection system. The methodology leverages an advanced ensemble approach, utilizing **Temporal Convolutional Networks (TCNs)** for time-series forecasting, and **Isolation Forests** combined with **Autoencoders** for anomaly detection. 

To ensure the model is robust regardless of physical environment, we fuse anomaly streams originating from three distinct environments:
- **Indoor Lab**
- **Industrial Hall**
- **Smart Building**

These three independent analysis streams are fed into a global **XGBoost Classifier** that generates a reliable, consolidated fire alarm risk probability, minimizing the chances of both false positive alarms and missed events across diverse internet-of-things deployments.

---

## Directory Structure & System Architecture

The project is modularized into two distinct workflows: the Jupyter Notebook development phase, and the active production deployment code.

### 1. Research & Model Development (`/notebooks`)
This module contains the research, training, testing, and comparative visualizations of the machine learning structures.
*   **Dataset 1, 2, 3 Notebooks:** End-to-end pipelines scaling from basic lab sensors (Temperature, Humidity, TVOCs) to heavy industrial gases and particulate matter (PM2.5, PM10, CO, CNG, LPG). 
*   **Generalized Risk Evaluation:** Dynamically assigns labels to unclassified telemetry features (using 95th-percentile thresholds) and maps data drifts. Generates correlation matrices, PR/ROC curves, and calculates confusion matrices to prove the advantage of the XGBoost fusion layer.
*(For deeper technical insights, refer to [Notebooks README](notebooks/README.md))*

### 2. Live Fusion Deployment (`/notebooks/models/fusion/fusion Model`)
This branch wraps up the generated `.pth`/`.pkl` model parameters into an executable framework designed for live monitoring (capable of running headlessly on devices like a Raspberry Pi).
*   **Generalized Processors:** Custom Python classes maintaining sliding data windows over live MQTT/Firebase streams. Data passes through scalers, TCN forecasters, and Autoencoders in real-time.
*   **Flask Aggregator Daemon:** A localized web server that listens asynchronously to a Firebase Realtime Database. As new telemetry drops into `/fireSensors/current`, inferences are made across all three environment models, the fused XGBoost decision is calculated, and the subsequent alarm status is reported to an included HTML web dashboard.
*(For full service runtime instructions, refer to [Live Deployment README](notebooks/models/fusion/fusion%20Model/README.md))*

### 3. Edge Node Firmware (`/final_update`)
This directory contains the C++ hardware sketch (`final_update.ino`) necessary to run the physical edge sensor node. It is designed to be deployed onto an ESP32 microcontroller and operates as the primary data ingestion layer.
*   **Hardware Supported:** MQ Gas Sensors (MQ2, MQ4, MQ6, MQ9), SDS011 Particulate Sensor, SHT31 (Temp/Humidity), SGP30 (TVOC, eCO2), and GUVA UV sensor.
*   **Networking & Sync:** Leverages `Firebase_ESP_Client` to connect over Wi-Fi and push telemetry data securely to the Firebase Realtime Database at `\fireSensors\current` on a sliding window interval, triggering inference in the daemonized Flask app.

---

## Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/YB-8080/capstone.git
cd capstone
```

2. **Establish Environment**
Ensure you are running **Python 3.8+**. Creating a virtual environment is highly recommended.
```bash
python -m venv .venv
# Activate environment on Windows:
.venv\Scripts\activate
# Activate environment on Linux/macOS:
source .venv/bin/activate
```

3. **Install Dependencies**
Install standard libraries via the included requirements package:
```bash
pip install -r requirements.txt
```

4. **Firebase Real-Time Capabilities** (Optional for Notebooks, Required for Deployment)
To drive the deployment models using actual data, a `serviceAccountKey.json` configuration file extracted from a Firebase Service Account must be placed into the backend inference folders. **Always keep `.gitignore` updated to prevent these keys from being exposed online.**

## Core Technology Stack
- **Deep Learning Pipelines:** `PyTorch` (`torch`, `torchvision`)
- **Traditional ML & Logic:** `scikit-learn`, `xgboost` 
- **Data Engineering:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Deployment APIs:** `Flask`, `firebase-admin`
