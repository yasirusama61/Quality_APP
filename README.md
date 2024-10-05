# Quality_APP with MES Integration

## Overview

The **Quality_APP** is designed for real-time monitoring and prediction of high and low-risk battery cells in a production line. By utilizing IPQC (In-Process Quality Control), FQC (Final Quality Control), and OQC (Outgoing Quality Control) data, the system predicts the risk associated with battery cells and provides recommendations for further actions:
- **High Risk**: Short-term testing recommended.
- **Low Risk**: Safe for all usage.

The application integrates with a Manufacturing Execution System (MES) to receive production orders and push data for quality analysis, ensuring comprehensive monitoring and control over the battery cell manufacturing process.

## Key Features
- Real-time monitoring of machine operational data (e.g., assembly force, pressure, speed) during battery cell manufacturing.
- Communication with MES to push machine data and fetch production orders.
- Quality control using machine learning models to predict defects.
- Corrective feedback sent to machines based on analysis.
- Dash-based web app for monitoring machine status, production orders, and analysis results.

## Features
- **Real-time Data Collection**: Collects quality control data (IPQC, FQC, OQC) from various stages of production.
- **Risk Prediction**: Machine learning models are used to predict high and low-risk battery cells.
- **Recommendation System**: Provides actionable insights, suggesting testing for high-risk cells and safe usage for low-risk cells.
- **MES Integration**: Communicates with MES via REST API for data exchange (production orders, quality control results).
- **Web Dashboard**: Visualizes production data and risk analysis results, facilitating real-time decision-making.

## Architecture Overview

### Data Sources:
- IPQC, FQC, and OQC data.
- MES API for integrating production orders and reporting quality control data.

### Core Functionality:

- Real-time data collection from machines.
- Analysis using machine learning models (Logistic Regression, XGBoost).
- Feedback loop to adjust machine parameters for quality control.

### Web Interface:

- A Dash-based interface for visualizing machine status and production orders.
- Interval-based updates to reflect the latest data and analysis.

### Requirements

- Python 3.8+
- Dash for web visualization
- Plotly for interactive charts
- Requests for communication with MES API
- scikit-learn for machine learning models
- XGBoost for model training
- MES API Access (replace with your actual MES URL and API token)

## Installation

### Clone the repository:

- `git clone https://github.com/yasirusama61/Quality_APP_MES_Integration.git`
- `cd Quality_APP_MES_Integration`

### Install the required dependencies:
- `pip install -r requirements.txt`

### requirements.txt:

- `dash==2.1.0`
- `dash-bootstrap-components==0.13.0`
- `requests==2.26.0`
- `pandas==1.3.3`
- `plotly==5.4.0`
- `scikit-learn==0.24.2`
- `xgboost==1.4.2`

## Usage

### Running the App Locally

Ensure you have your MES API token set up in the environment variables or update the MES_API_URL and HEADERS in the script.

Run the following command to start the app:

- `python app.py`

Access the Dash app in your browser at: `http://127.0.0.1:8050`.

### Running the App in Production

Use Gunicorn or a similar WSGI server for production deployment.


- `gunicorn -b 0.0.0.0:8050 app:app`

Make sure the app is properly configured to interact with the actual production line machines and the MES API.

### Interacting with the Dashboard

- **Machine Monitoring Section**: Displays real-time machine data such as temperature, vibration, speed, and status.
- **Production Orders Section**: Shows active production orders fetched from the MES.
- **Risk Prediction and Recommendation**: Provides real-time predictions of high/low-risk battery cells based on IPQC, FQC, and OQC data, with recommendations on further testing or clearance.

### Integration Details

### Machine Data Fetching
Machine data is simulated in the `get_machine_data()` function. In a real-world environment, replace this with actual data from your PLCs, IoT devices, or sensors.
The data includes critical operational parameters such as temperature, speed, vibration, and machine status.
MES API Communication
The app sends real-time machine data to the MES using `post_machine_data_to_mes()` and retrieves production orders using fetch_production_orders().
You need to replace the placeholder URL `(MES_API_URL)` and API token in the script with actual values for your MES system.

### Machine Learning Models

The `analyze_quality()` function simulates a quality control model to predict potential defects based on machine operating parameters.
You can replace this with actual machine learning models, such as Logistic Regression or XGBoost, trained on your historical production data.

### Feedback to Machines
Based on the results of the quality analysis, corrective feedback is sent to machines via the `send_correction_to_machine()` function.In real-world use cases, this would involve sending signals to the machine's PLC to adjust operating parameters (e.g., speed, temperature).

### Usage

- Run the script in a local environment or deploy it on a server.
- View real-time machine data and production orders from the MES in the Dash web app.
- Analyze machine data for defects using machine learning models, and see the corrective feedback sent to machines.

### Expected Output

### Machine Status:

- **Machine ID**: CAS_01 (Cell Assembly Station)
  - **Temperature**: 45°C
  - **Vibration**: 0.12 m/s²
  - **Speed**: 1000 RPM
  - **Status**: Operating

- **Machine ID**: FORM_01 (Formation Chamber)
  - **Temperature**: 38°C
  - **Current**: 3.5 A
  - **Voltage**: 4.2 V
  - **Status**: Charging

- **Machine ID**: EOL_01 (End-Of-Line Tester)
  - **Temperature**: 40°C
  - **Voltage**: 3.9 V
  - **SOC**: 98%
  - **Status**: Testing

- **Production Orders**:
  - Order 98765: High Capacity Cell - 2000 units
  - Order 98766: Standard Capacity Cell - 1500 units

### Defect Analysis:

- **Defect Predicted**: Yes
- **Corrective Action Sent**: Reduce Speed

### Future Enhancements

- **Advanced Machine Learning Models**: Integrate models like Random Forest, Neural Networks, or LSTMs for more robust defect prediction.
- **Data Persistence**: Store machine data and quality analysis results in a database (e.g., PostgreSQL, MongoDB) for further reporting and analytics.
- **Real Machine Data Integration**: Replace the simulated machine data with actual production line data from PLC controllers, IoT devices, or OPC servers.
- **Full MES Integration**: Improve communication with the MES to handle more complex tasks such as job scheduling, downtime tracking, and machine status alerts.