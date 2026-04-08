# Multi-Model Fire Risk Fusion System

This folder contains the consolidated deployment architecture for the generalized internet-of-things fire detection system. Instead of relying on a single model or sensor type, this system ensembles predictions from three disparate physical environments (Indoor Lab, Industrial Hall, and Smart Building) to generate a highly robust and reliable global fire risk alarm.

## Directory Structure

*   **/TRAINED MODELS**: Contains the exported PyTorch TCN Forecasters, Autoencoders, Scikit-learn Isolation Forests, Data Scalers, and XGBoost predictors compiled from all three dataset environments.
*   **/MAIN CODES**: Contains the executable Python applications and local web interface (`index_final_fusion.html`) that utilize the trained models.

## Core Components

1.  **Generalized Fire Predictors (`multi_predict_final_fusion.py`)**: End-to-end classes that establish sliding windows over incoming telemetry, pass data through Temporal Convolutional Networks to generate reconstruction residuals, and rate the anomaly status via Isolation Forests and Autoencoders.
2.  **Global XGBoost Combiner (`train_global_fusion_realdata.py`)**: Script used to train the top-level XGBoost fusion tree across the three disparate model streams. Optimizes recall to minimize false negatives (missed alarms).
3.  **Flask Deployment App (`corrected_deployment_app_fusion.py`)**: The primary server application. It daemonizes a worker thread that listens to a Firebase Realtime Database for new telemetry data. It pushes incoming sensor data through all three inference pipelines to determine if a global fire alarm threshold is met, updating a local web dashboard and writing results back to Firebase in real time.

## Running the Deployment System

### 1. Install Dependencies
Make sure you have your Python environment activated, then install the necessary packages using the requirements file included in this directory:

```bash
pip install -r requirements.txt
```

### 2. Configure Firebase
Ensure your Firebase `serviceAccountKey.json` from the Google Cloud Console is placed inside the `MAIN CODES` directory, and verify that the `FIREBASE_DB_URL` inside `corrected_deployment_app_fusion.py` matches your Realtime Database.

### 3. Launch the Application
Run the Flask server:
```bash
cd "MAIN CODES"
python corrected_deployment_app_fusion.py
```

The system will report its warm-up status as it begins collecting multi-sensor telemetry blocks from Firebase (`/fireSensors/current`) and will start broadcasting inference results to `http://localhost:5000/api/latest` and Firebase (`/fireRisk/current`).
