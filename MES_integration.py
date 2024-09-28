import requests
import json
import time
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import random

# Simulating real-time data from machines on the production line
def get_machine_data():
    """
    Simulate fetching real-time data from a machine on the production line.
    In a real-world scenario, this would involve connecting to PLCs, IoT sensors, or OPC servers.
    """
    return {
        'machine_id': 'CNC_01',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': random.uniform(50.0, 120.0),  # Simulating temperature
        'vibration': random.uniform(0.1, 1.5),  # Simulating vibration levels
        'speed': random.uniform(500, 1500),  # Simulating machine speed in RPM
        'status': 'operating' if random.random() > 0.1 else 'fault',  # Random machine status
    }

# MES API endpoint for sending and receiving data
MES_API_URL = "https://mes.example.com/api/production_data"
HEADERS = {
    'Authorization': 'Bearer YOUR_API_TOKEN',  # Add proper API token
    'Content-Type': 'application/json'
}

# Function to push machine data to MES
def post_machine_data_to_mes(machine_data):
    try:
        response = requests.post(MES_API_URL, headers=HEADERS, data=json.dumps(machine_data))
        if response.status_code == 200:
            print("Machine data successfully sent to MES.")
        else:
            print(f"Failed to send machine data to MES: {response.status_code}")
    except Exception as e:
        print(f"Error sending machine data to MES: {e}")

# Fetch production orders from MES (could be product type, quantity, machine setup, etc.)
def fetch_production_orders():
    try:
        response = requests.get(MES_API_URL + '/orders', headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch production orders from MES: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to MES: {e}")
        return None

# Function to perform quality analysis (simulating Logistic Regression, XGBoost, etc.)
def analyze_quality(machine_data):
    """
    Simulate a machine learning model that predicts if a machine will produce defective products
    based on machine operating parameters (temperature, vibration, speed, etc.).
    """
    # Simulate defect probability based on operating parameters
    defect_probability = 0.05 * machine_data['temperature'] + 0.2 * machine_data['vibration'] + 0.1 * machine_data['speed']
    
    # If defect probability exceeds a threshold, mark it as a potential defect
    return defect_probability > 1000

# Feedback to the machine based on analysis (e.g., adjust speed or temperature)
def send_correction_to_machine(machine_id, correction):
    """
    In a real-world scenario, this function would send signals back to the machine's PLC or controller
    to adjust parameters like speed or temperature.
    """
    print(f"Sending correction to {machine_id}: {correction}")

# Simulate data collection, analysis, and feedback loop
def run_production_line_integration():
    # Get machine data in real-time
    machine_data = get_machine_data()

    # Send machine data to MES
    post_machine_data_to_mes(machine_data)

    # Analyze quality of production based on machine data
    is_defective = analyze_quality(machine_data)

    if is_defective:
        # Send corrective feedback to the machine (e.g., slow down speed or reduce temperature)
        send_correction_to_machine(machine_data['machine_id'], correction="Reduce speed")

    # Fetch production orders from MES (optional)
    production_orders = fetch_production_orders()

    # Return data for visualization
    return machine_data, production_orders

# Dash app for monitoring production line and MES integration
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Production Line - Machine and MES Monitoring"),
    dcc.Interval(id='interval-component', interval=2*1000, n_intervals=0),  # Refresh every 2 seconds
    html.Div(id="machine-status"),
    html.Div(id="mes-orders"),
])

# Callback to update dashboard with real-time machine data and MES orders
@app.callback(
    [Output('machine-status', 'children'),
     Output('mes-orders', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Get real-time machine data and production orders
    machine_data, production_orders = run_production_line_integration()

    # Display machine data
    machine_status = html.Div([
        html.H3(f"Machine {machine_data['machine_id']} Status: {machine_data['status']}"),
        html.P(f"Temperature: {machine_data['temperature']} °C"),
        html.P(f"Vibration: {machine_data['vibration']} m/s²"),
        html.P(f"Speed: {machine_data['speed']} RPM"),
        html.P(f"Timestamp: {machine_data['timestamp']}")
    ])

    # Display production orders from MES
    if production_orders:
        mes_orders = html.Div([
            html.H3("Production Orders from MES"),
            html.Ul([html.Li(f"Order {order['order_id']}: {order['product']} - {order['quantity']} units") for order in production_orders])
        ])
    else:
        mes_orders = html.Div(html.P("No production orders found."))

    return machine_status, mes_orders

# Run the app in production mode
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
