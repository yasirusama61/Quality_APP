import requests
import json
import time
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sklearn.externals import joblib  # For loading pre-trained models

# MES API endpoint for fetching battery manufacturing data
MES_API_URL = "https://mes.example.com/api/production_data"  # Replace with actual MES API

# Features used for risk level prediction
features = [
    'positive_electrode_viscosity', 'negative_electrode_viscosity', 'electrode_coating_weight',
    'electrode_thickness', 'electrode_alignment', 'welding_bead_size', 'lug_dimensions', 
    'moisture_content_after_baking', 'electrolyte_weight', 'formation_energy', 
    'aging_time', 'pressure', 'ambient_temperature'
]

# Load pre-trained risk prediction model (assumes model is already trained and saved as a .pkl file)
model = joblib.load('risk_prediction_model.pkl')

# Function to fetch real-time data from MES
def fetch_battery_data_from_mes():
    """
    Fetch real-time data for the battery manufacturing process from the MES.
    In a real-world scenario, this would involve connecting to the MES API or a database.
    """
    try:
        response = requests.get(MES_API_URL, headers={'Authorization': 'Bearer YOUR_API_TOKEN'})  # Replace with your actual token
        if response.status_code == 200:
            return response.json()  # Assuming the response contains the required features as JSON
        else:
            print(f"Failed to fetch data from MES: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to MES: {e}")
        return None

# Function to predict risk level based on features
def predict_risk_level(data):
    """
    Predict the risk level based on the provided data.
    :param data: Dictionary containing feature values from MES.
    :return: Predicted risk level.
    """
    # Extract features from the data
    input_data = [data[feature] for feature in features]

    # Convert input data to DataFrame (since model might expect it as a DataFrame)
    input_df = pd.DataFrame([input_data], columns=features)

    # Predict risk level using the pre-trained model
    risk_level = model.predict(input_df)[0]  # Assuming classification model
    return risk_level

# Dash app for monitoring and risk prediction
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Battery Manufacturing - Real-time Risk Prediction"),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),  # Refresh every 5 seconds
    html.Div(id="battery-data"),
    html.Div(id="risk-prediction"),
])

# Callback to update dashboard with real-time data and risk prediction
@app.callback(
    [Output('battery-data', 'children'),
     Output('risk-prediction', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch real-time battery manufacturing data from MES
    battery_data = fetch_battery_data_from_mes()

    if battery_data:
        # Predict risk level
        risk_level = predict_risk_level(battery_data)

        # Display battery data and risk level on the dashboard
        battery_info = html.Div([
            html.H3(f"Battery Data (Machine ID: {battery_data['machine_id']})"),
            html.P(f"Positive Electrode Viscosity: {battery_data['positive_electrode_viscosity']}"),
            html.P(f"Negative Electrode Viscosity: {battery_data['negative_electrode_viscosity']}"),
            html.P(f"Electrode Coating Weight: {battery_data['electrode_coating_weight']} g"),
            html.P(f"Electrode Thickness: {battery_data['electrode_thickness']} mm"),
            html.P(f"Welding Bead Size: {battery_data['welding_bead_size']} mm"),
            html.P(f"Lug Dimensions: {battery_data['lug_dimensions']} mm"),
            html.P(f"Moisture Content After Baking: {battery_data['moisture_content_after_baking']}%"),
            html.P(f"Electrolyte Weight: {battery_data['electrolyte_weight']} g"),
            html.P(f"Formation Energy: {battery_data['formation_energy']} kWh"),
            html.P(f"Aging Time: {battery_data['aging_time']} hours"),
            html.P(f"Pressure: {battery_data['pressure']} Pa"),
            html.P(f"Ambient Temperature: {battery_data['ambient_temperature']} °C"),
        ])

        risk_info = html.Div([
            html.H3(f"Predicted Risk Level: {risk_level}"),
            html.P("High Risk" if risk_level == 1 else "Low Risk")
        ])
    else:
        battery_info = html.Div("Failed to fetch battery data.")
        risk_info = html.Div("Risk level prediction not available.")

    return battery_info, risk_info

# Run the app in production mode
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
