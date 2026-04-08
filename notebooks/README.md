# Fire Risk Detection Notebooks

This folder contains the Jupyter notebooks used to develop, train, and evaluate the machine learning pipelines for fire risk detection across three different environments.

## Overview

The methodology leverages timeseries forecasting (Temporal Convolutional Networks) and anomaly detection (Isolation Forests and Autoencoders) on multi-sensor data to generate a fire risk index. For complex environments, these anomaly streams are fused using an XGBoost Classifier to output final event probabilities.

### Notebooks

*   **`01_dataset1_ssl_tcn.ipynb`**
    *   **Focus:** Dataset 1 (Indoor Lab) Base Models.
    *   **Description:** Preprocesses data from multiple sensors (Temperature, Humidity, TVOC, eCO2, MQ139). Trains a TCN Forecaster to establish a baseline and computes "Residual-Based Fire Risk Signals". Trains an Isolation Forest and a Residual Autoencoder for further anomaly detection.
*   **`02_dataset1_risk_analysis.ipynb`**
    *   **Focus:** Dataset 1 Risk Evaluation & Smoothing.
    *   **Description:** Applies the trained models to all test scenarios to output a consolidated Risk score. Introduces and tunes a custom `persistence_filter` to smooth risk spikes and evaluate the precision/recall of the alarm trigger system.
*   **`03_dataset2_industrial_hall.ipynb`**
    *   **Focus:** Dataset 2 (Industrial Hall) Full Pipeline.
    *   **Description:** Adapts the architecture for industrial features (PM25, PM10, CO2_Room, VOCs). Extracts residual and anomaly scores from a TCN, Isolation Forest, and Autoencoder. Fuses these features using an `XGBClassifier` to generate the final fire prediction probabilities.
*   **`04_dataset3_smart_building.ipynb`**
    *   **Focus:** Dataset 3 (Smart Building) Labeling & Fusion.
    *   **Description:** Creates ground truth labels dynamically using 95th percentile-based multi-sensor rules (CO, CNG, LPG, Smoke, Flame). Trains the complete pipeline (TCN -> Isolation Forest -> Autoencoder), calculates data drift, and uses an `XGBClassifier` over these features for risk modeling and alarm generation.
*   **`05_generalized_fire_risk_model.ipynb`**
    *   **Focus:** Cross-Dataset Generalization.
    *   **Description:** Consolidates the risk indices and residual predictions from all three environments. Generates comparative visualizations (boxplots and bar charts) to evaluate the generalized performance and robustness of the methodology across different physical scales and sensor layouts.

## Dependencies & Requirements

To run these notebooks and the associated pipeline, you will need **Python 3.8+** and the following core libraries:

*   **Data Manipulation & Analysis:** `pandas`, `numpy`
*   **Machine Learning (Core):** `scikit-learn`, `xgboost`
*   **Deep Learning:** `torch` (PyTorch), `torchvision`, `torchaudio`
*   **Visualization:** `matplotlib`, `seaborn`
*   **File I/O:** `openpyxl` (required for parsing `.xlsx` datasets)

You can install these dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn xgboost torch torchvision torchaudio matplotlib seaborn openpyxl
```

## Workflow

To follow the development lifecycle, it's recommended to run the notebooks sequentially:
1. Start with Dataset 1 to understand the core anomaly-generation logic.
2. Progress to Datasets 2 & 3 to see how the single-model signals are generalized into the XGboost fusion layer.
3. Review Notebook 5 to see final comparative metrics.
