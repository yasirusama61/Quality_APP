import requests
import json
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from googletrans import Translator
import pdfkit
from flask import send_file
import time
import plotly.io as pio

# Load pre-trained model
model = xgb.Booster()
model.load_model("pretrained_model.json")  # Path to the pre-trained model

# Assuming used a scaler during training, load it similarly
scaler = StandardScaler()  # Assuming scaler is saved, load it accordingly

# MES API endpoint for fetching battery manufacturing data
MES_API_URL = "https://mes.example.com/api/production_data"  # Replace with actual MES API URL
headers = {'Authorization': 'Bearer YOUR_API_TOKEN'}  # Replace with actual API token

app = dash.Dash(__name__, 
           external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'], 
           suppress_callback_exceptions=True)

translator = Translator()

# Translation function
def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# Fetch battery manufacturing data from MES
def fetch_battery_data():
    try:
        response = requests.get(MES_API_URL, headers=headers)
        if response.status_code == 200:
            return response.json()  # Assuming the response contains the required features as JSON
        else:
            print(f"Failed to fetch data from MES: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to MES: {e}")
        return None

app.layout = html.Div([
    html.H1(translate_text("Model Overview", 'en'), id='model-overview-heading', style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='language-dropdown',
                options=[
                    {'label': 'English', 'value': 'en'},
                    {'label': '中文', 'value': 'zh-TW'},
                ],
                value='en',
                clearable=False,
                style={
                    'width': '90%',
                    'margin': '10px',
                    'border': '1px solid #ccc',
                    'border-radius': '5px',
                },
                className='dropdown-style'
            ),
        ]),
    ]),

    html.Div(id='welcome-message-container', children=[
        html.H3(translate_text("Welcome to the Model Overview Dashboard", 'en'), style={'padding': '20px'}),
        html.P(translate_text("This dashboard provides an overview of the model and allows you to perform various tasks.", 'en'),
               style={'padding': '20px'}),
        html.P(translate_text("Follow the steps below to use the dashboard:", 'en'),
               style={'padding': '20px'}),
        html.Ul([
            html.Li(translate_text("Step 1: Fetch MES data and run the model.", 'en')),
            html.Li(translate_text("Step 2: View risk prediction and download the results.", 'en')),
        ]),
    ], style={'margin-bottom': '20px'}),

    html.Div([
        dbc.Button("Execute", id="execute-button", style={'margin-bottom': '20px', 'margin-left': '20px'}, color='primary'),
        dcc.Loading(id="loading-prediction", type="circle", children=[html.Div(id='execute-output')]),
    ]),

    html.Div([
        html.H3("Step 2: Download Report", style={'font-weight': 'bold', 'margin-left': '20px', 'margin-top': '20px'}),
        dbc.Button("Download Report", id="btn-download", color="primary", className="mb-3", style={'margin-left': '20px'}),
        dcc.Download(id="download")
    ]),

    # Placeholder for plots
    html.Div([
        html.H3("Confusion Matrix", style={'font-weight': 'bold', 'margin-left': '20px'}),
        dcc.Graph(id='confusion-matrix-plot'),

        html.H3("Risk Distribution", style={'font-weight': 'bold', 'margin-left': '20px'}),
        dcc.Graph(id='risk-distribution-plot')
    ])
])

# Callback to handle language switching and text translation
@app.callback(
    [Output('model-overview-heading', 'children'),
     Output('welcome-message-container', 'children')],
    [Input('language-dropdown', 'value')],
)
def update_text(selected_language):
    translated_model_heading = translate_text("Model Overview", selected_language)

    translated_welcome_message = [
        html.H3(translate_text("Welcome to the Model Overview Dashboard", selected_language), style={'padding': '20px'}),
        html.P(translate_text("This dashboard provides an overview of the model and allows you to perform various tasks.", selected_language),
               style={'padding': '20px'}),
        html.P(translate_text("Follow the steps below to use the dashboard:", selected_language),
               style={'padding': '20px'}),
        html.Ul([
            html.Li(translate_text("Step 1: Fetch MES data and run the model.", selected_language)),
            html.Li(translate_text("Step 2: View risk prediction and download the results.", selected_language)),
        ]),
    ]

    return translated_model_heading, translated_welcome_message

# Callback to handle MES data fetching, risk prediction, and plotting
@app.callback(
    [Output('execute-output', 'children'),
     Output('confusion-matrix-plot', 'figure'), Output('risk-distribution-plot', 'figure')],
    [Input('execute-button', 'n_clicks')],
    prevent_initial_call=True
)
def predict_risk_level(n_clicks):
    # Fetch real-time data from MES
    mes_data = fetch_battery_data()
    
    if mes_data is None:
        return "Failed to fetch data from MES.", dash.no_update, dash.no_update

    try:
        # Preprocess the data
        X = pd.DataFrame([mes_data])  # Convert MES data to DataFrame
        X_scaled = scaler.transform(X)  # Normalize the data

        # Use the pre-trained model to predict
        dmatrix = xgb.DMatrix(X_scaled)
        y_pred = model.predict(dmatrix)
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

        # Confusion Matrix and Distribution Plot
        low_risk_count = y_pred_binary.count(0)
        high_risk_count = y_pred_binary.count(1)

        # Generate a pie chart of risk distribution
        risk_fig = go.Figure(data=[go.Pie(labels=['Low Risk', 'High Risk'],
                                          values=[low_risk_count, high_risk_count],
                                          hole=.3)])
        risk_fig.update_layout(title_text='Risk Level Distribution')

        # Generate a dummy confusion matrix (replace with real values if available)
        conf_matrix_fig = go.Figure(data=go.Heatmap(
            z=[[50, 10], [5, 35]],  # Example confusion matrix values
            x=['Predicted Low Risk', 'Predicted High Risk'],
            y=['Actual Low Risk', 'Actual High Risk'],
            colorscale='Viridis'
        ))
        conf_matrix_fig.update_layout(title_text='Confusion Matrix')

        prediction_text = f"Predicted risk levels: {y_pred_binary}"
        return prediction_text, conf_matrix_fig, risk_fig

    except Exception as e:
        return f"Error: {str(e)}", dash.no_update, dash.no_update

# Generate PDF Report
def generate_report(prediction_data, risk_fig, conf_matrix_fig, file_name="report.pdf"):
    html_content = f"""
    <html>
    <body>
        <h1>Risk Prediction Report</h1>
        <p>Here are the results of your prediction:</p>
        <p><strong>Predicted Risk Levels:</strong> {prediction_data}</p>
        <hr>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>Risk Distribution</h2>
        <img src="data:image/png;base64,{pio.to_image(risk_fig, format='png').decode('utf-8')}" />
        <h2>Confusion Matrix</h2>
        <img src="data:image/png;base64,{pio.to_image(conf_matrix_fig, format='png').decode('utf-8')}" />
    </body>
    </html>
    """
    
    # Save the PDF
    pdfkit.from_string(html_content, file_name)
    return file_name

# Callback to generate and download the report
@app.callback(
    Output("download", "data"),
    [Input("btn-download", "n_clicks"), Input('execute-output', 'children'),
     Input('risk-distribution-plot', 'figure'), Input('confusion-matrix-plot', 'figure')],
    prevent_initial_call=True
)
def download_report(n_clicks, prediction_output, risk_fig, conf_matrix_fig):
    if n_clicks is not None and prediction_output:
        pdf_file = generate_report(prediction_output, risk_fig, conf_matrix_fig)
        return dcc.send_file(pdf_file)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
