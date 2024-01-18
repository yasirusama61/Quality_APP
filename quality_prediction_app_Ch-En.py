import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from io import BytesIO
import base64
import time
from datetime import datetime
import dash_bootstrap_components as dbc
from dash import dcc, html
from IPython.display import HTML
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from dash import Dash, dcc, html
import os
import plotly.io as pio
import zipfile
import shutil
import mimetypes
from dash import callback_context
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import mimetypes
import binascii
from reportlab.lib.utils import ImageReader
from PIL import Image
from flask import send_file
import io
import json
import pdfkit
import plotly.express as px
from sklearn.svm import SVC
import xgboost as xgb
from datetime import datetime as dt
import ipywidgets as widgets
from IPython.display import display
from plotly.io import to_image
from io import BytesIO
import tempfile
from fpdf import FPDF
from plotly.offline import plot
from dash import html
import plotly
from weasyprint import HTML
import webbrowser
from plotly.io import to_html
import pdfkit
from plotly.io import write_image
import subprocess
from datetime import timedelta
from googletrans import Translator


# Load your data
loaded_data = None
model = None
conf_matrix_fig = None  
fig = None          
fig_table = None           
fig_metrics_table = None
figure_train = None
lot_score_percentage = 100.0
conf_matrix_store_data = None
fig_store_data = None
fig_table_store_data = None
fig_metrics_table_store_data = None
figure_train_store_data = None
start_time = time.time()


# Create a text input widget for the author's name
author_name_widget = widgets.Text(description="Author's Name:")
display(author_name_widget)


app = Dash(__name__, 
    external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'], 
           suppress_callback_exceptions=True)

pdfkit_config = pdfkit.configuration(wkhtmltopdf='')

# Translation function
def translate_text(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

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
            html.Li(translate_text("Step 1: Upload your data using the 'Upload Data' button.", 'en')),
            html.Li(translate_text("Step 2: Run the model using the 'Execute' button.", 'en')),
            html.Li(translate_text("Step 3: Download the results.", 'en')),
        ]),
        html.Div("Step 1: Upload Data here", style={'font-weight': 'bold','margin-top': '20px' ,
                                                    'margin-bottom':'20px',
                                                     'padding': '20px'}),
    ], style={'margin-bottom': '20px'}),

    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button('Upload Data', "Success", color="success", className="me-1"),
                        style={'padding': '20px'},
                        multiple=False
                    ),
                    dbc.Button('Cancel', id='cancel-button', color='danger', className='me-2', n_clicks=0,
                               style={'margin-top': '10px', 'margin-left': '20px', 'margin-bottom': '20px'})
                ]),
                width={'size': 4, 'offset': 0}
            ),
        ]),
    ]),
    
    dcc.Loading(
        id="loading-upload",
        type="circle",
        children=[
            html.Div(id='upload-loading-output'),
        ],
        style={'margin-top': '20px'}
    ),
    
    # Message box to display upload status
    html.Div(id='upload-message'),

    # Message box to display attribute info
    html.Div(id='attribute-info', style={'color': 'green', 'margin-bottom': '10px'}),
    
    # Progress bar and button
    html.Div([
        html.Div("Step 2: Run the model", id = 'step-2-text', style={'font-weight': 'bold','margin-top': '20px' ,'margin-bottom':'20px',
                                                     'padding': '20px'}),
        dbc.Button("Execute", id="train-predict-button", style={'margin-bottom': '20px',
                                                                          'margin-left': '20px'}, color='primary'),
        html.Div(id="training-output"),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
        dbc.Progress(id="training-progress-bar", value=str(0), max=100,
              style={'width': '90%', 'height': '30px', 'textAlign': 'center', 'margin-left': '20px'},
              className='progress-bar-striped', color="success", striped=True, animated=True),
        html.Div(id="remaining-time", children="", style={'margin-top': '10px', 'margin-bottom': '20px'}),
        dcc.Store(id='start-time-store', data=None)
    ]),
    
    html.Div([
        html.H3(id='training-history-heading', style={'margin-left': '20px','margin-bottom': '20px' }),
        html.Div([
            dcc.Graph(id='fig-train', style={'width': '50%', 'margin-left': '10%'}),
            dcc.Markdown(id='figure-train-description', style={'width': '50%'}),
        ], style = {'display': 'flex', 'justify-content': 'center', 'margin-bottom':'20px'})
    ]),

    html.H3(id='confusion-matrix-heading', style={'margin-left': '20px'}),
    html.Div([
        dcc.Graph(id='confusion-matrix', style={'width': '50%','margin-left': '20px', 'display': 'flex',
                                               'justify-content':'center'}),
        html.Div([
            dcc.Graph(id='fig-metrics-table', style={'width': '90%'}),
            dcc.Markdown(id='conf-matrix-description', style={'width': '90%', 'margin-left':'23px'}),
        ], style={'width': '50%', 'display': 'inline'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom':'20px'}),

    html.Div([
        html.H3(id='risk-level-distribution-heading', style={'margin-left': '20px'}),
        html.Div([
            dcc.Graph(id='fig', style={'width': '50%', 'margin-left': '20px', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='fig-table', style={'width': '90%'}),
                dcc.Markdown(id='lot-risk-description', style={'width': '90%', 'margin-left': '20px'}),
            ], style={'width': '50%', 'display': 'inline', 'margin-top': '20px', 'margin-bottom': '20px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
    ]),
    
    dbc.Card(
            dbc.CardBody(
                [
                    html.H3("Lot Risk", id='lot-risk-card-title', 
                            style={'margin-top': '10px', 'margin-bottom': '10px'}),
                    dcc.Markdown(id='risk-scores-markdown', style={'font-size': '20px', 
                                                                   'font-weight': 'bold', 'color': 'black'}),
                    html.Hr(),  # Horizontal line for separation
                    html.H5("Recommended Action", id = 'recommendation-line',
                        style={'margin-top': '15px', 'margin-bottom': '10px'}),
                    html.Div(id='recommendation-output'),
                ]
            ),
            id='lot-risk-score-card', 
            style={'width': '10%', 'margin-left': '20px', 'display': 'none', 'border':'2px solid #000000', 
                   'border-radius': '5px', 'padding': '10px', 'margin-bottom':'20px','background-color': '#e0e0e0'}
    ),
    
    html.Div([
        html.Div("Step 3: Download the Report", id='step-3-text',
                 style={'font-weight': 'bold', 'margin-left': '20px', 'margin-bottom': '20px'}),
        html.Div([
            html.Div("***Click the button below to download the report in pdf format:", id='download-line',
               style={'margin-left': '20px', 'color': 'green', 'margin-bottom':'20px'}),
            dbc.Button(
                    "Download Report",
                    id="btn-download",
                    color="primary",
                    className="mb-3",
                    style={'margin-left': '20px'},  
            ),
            dcc.Download(id="download"),
        ]),
    ]),
    dcc.Store(id='conf-matrix-store'),
    dcc.Store(id='fig-store'),
    dcc.Store(id='fig-table-store'),
    dcc.Store(id='fig-metrics-table-store'),
    dcc.Store(id='figure-train-store'),
])

# Callback to handle language switching and text translation
@app.callback(
    Output('model-overview-heading', 'children'),
    Output('welcome-message-container', 'children'),
    # Add more Outputs for other elements want to translate
    Input('language-dropdown', 'value'),
)
def update_text(selected_language):
    translated_model_heading = translate_text("Model Overview", selected_language)
    #translated_button_text = translate_text("Translate", selected_language)

    # Translate the children of the welcome message container
    translated_welcome_message = [
        html.H3(translate_text("Welcome to the Model Overview Dashboard", selected_language), style={'padding': '20px'}),
        html.P(translate_text("This dashboard provides an overview of the model and allows you to perform various tasks.", selected_language),
               style={'padding': '20px'}),
        html.P(translate_text("Follow the steps below to use the dashboard:", selected_language),
               style={'padding': '20px'}),
        html.Ul([
            html.Li(translate_text("Step 1: Upload your data using the 'Upload Data' button.", selected_language)),
            html.Li(translate_text("Step 2: Run the model using the 'Execute' button.", selected_language)),
            html.Li(translate_text("Step 3: Download the results.", selected_language)),
        ]),
        html.Div(translate_text("Step 1: Upload Data here",selected_language),
                 style={'font-weight': 'bold','margin-top': '20px' ,'margin-bottom':'20px','padding': '20px'}),
    ]

    return translated_model_heading, translated_welcome_message  


# Callbacks
@app.callback(
    [Output("upload-data", "children"),
     Output("upload-message", "children"),
     Output("attribute-info", "children"),
     Output("upload-loading-output", "children"),
     Output("cancel-button", "children"),
     Output('step-2-text', 'children'),
    Output("train-predict-button", "children"),
    Output('step-3-text', 'children'), 
    Output('download-line', 'children'),
    Output('btn-download','children'), 
    Output('lot-risk-card-title', 'children'),
    Output('recommendation-line','children')],
    [Input('language-dropdown', 'value'),
     Input('upload-data', 'contents')],
)

def translate_components(selected_language, contents):
    global loaded_data
    if selected_language is None:
        raise dash.exceptions.PreventUpdate

    # Translate buttons and messages based on the selected language
    translated_upload_button_text = translate_text('Upload Data', selected_language)
    translated_cancel_button_text = translate_text('Cancel', selected_language)
    translated_upload_success_message = translate_text('File uploaded successfully!', selected_language)
    translated_step_2_text = translate_text('Step 2: Run the model', selected_language)
    translated_execute_button_text = translate_text('Execute', selected_language)
    translated_step_3_text = translate_text('Step 3: Download the Report', selected_language)
    translated_download_line = translate_text('***Click the button below to download the report in pdf format:',
                                             selected_language)
    translated_download_button_text = translate_text('Download Report', selected_language)
    translated_card_title = translate_text('Lot Risk', selected_language)
    translated_recommendation_line = translate_text('Recommended Action', selected_language)

    # Update button labels with translated text
    upload_button = dbc.Button(translated_upload_button_text, "Success", color="success", className="me-1")
    cancel_button = dbc.Button(translated_cancel_button_text, 'Cancel', color='danger', className='me-2')
    execute_button = dbc.Button(translated_execute_button_text, 'Execute',color='primary')

    # If no file is uploaded, return translation and prevent data loading
    if contents is None:
        return upload_button, dash.no_update, dash.no_update, dash.no_update, cancel_button, \
        translated_step_2_text,execute_button,translated_step_3_text,translated_download_line, \
        translated_download_button_text,translated_card_title,translated_recommendation_line

    # Assume the uploaded file is in Excel format
    content_type, content_string = contents.split(',')    ##cancel_button, \
    decoded = base64.b64decode(content_string)

    try:
        loaded_data = pd.read_excel(BytesIO(decoded), engine='openpyxl')

        # Simulate some processing time (we can replace this with our actual processing logic)
        # For demonstration purposes, sleep for 3 seconds.
        time.sleep(3)

        # Display message with the number of attributes and their shape
        translated_attributes_message = translate_text(
            f"Number of attributes: {len(loaded_data.columns)}, Number of rows: {len(loaded_data)}",
            selected_language
        )

        attribute_info = dbc.Alert(
            translated_attributes_message,
            color="info", style={'margin-left': '20px', 'width': '90%'})
        
        #remaining_time_text = "Expected Time:"
        #translated_remaining_time = translate_text(f"Expected Time: {remaining_time_text}", selected_language)

        return upload_button, dbc.Alert(translated_upload_success_message, style={'color': 'green', 'margin-left': '20px',
                                                                                  'margin-bottom': '10px', 'width': '90%'}), \
               attribute_info, dash.no_update, cancel_button, translated_step_2_text, execute_button, \
               translated_step_3_text,translated_download_line,translated_download_button_text, \
               translated_card_title, translated_recommendation_line  

    except Exception as e:
        loaded_data = None
        error_message = dbc.Alert(f'Error uploading file: {str(e)}', color='danger')
        return upload_button, error_message, dash.no_update, dash.no_update, cancel_button, translated_step_2_text, \
        execute_button, translated_step_3_text,translated_download_line,translated_download_button_text, \
        translated_card_title, translated_recommendation_line  #cancel_button,


# Callback to handle both training/predicting and downloading
@app.callback(
    [Output("training-progress-bar", "value"),
     Output("remaining-time", "children")],
    [Input('train-predict-button', 'n_clicks'),
     Input('cancel-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State("start-time-store", "data")],
    prevent_initial_call=True, 
)

def train_and_predict(n_clicks, cancel_clicks, n_intervals, start_time_data):
    global start_time, start_time_first_iteration, progress_values

    if cancel_clicks and cancel_clicks > 0:
        clear_loaded_data()
        return 0, html.Div("Data cleared.", style={'margin-left': '20px', 'font-weight': 'bold'})

    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
        
    if data_is_not_loaded():
        error_message = "Error: Data not loaded. Please upload data first."
        return error_message, str(0)

    epochs = 300

    if n_intervals == 0:
        start_time_first_iteration = time.time()

    elapsed_time = time.time() - start_time
    current_step = n_intervals

    # Calculate overall progress based on completion of each epoch
    overall_progress = len(progress_values) / epochs * 100

    if overall_progress >= 100:
        remaining_time = 0
    else:
        average_time_per_iteration = elapsed_time / current_step if current_step > 0 else 0
        remaining_epochs = epochs - current_step
        remaining_time = remaining_epochs * average_time_per_iteration
        remaining_time = max(remaining_time, 0)

    expected_time_str = f"Expected Time: {str(timedelta(seconds=round(remaining_time)))}"
    progress_values.append(overall_progress)  # Add progress to the list

    return [int(overall_progress),html.Div(expected_time_str, style={'margin-left': '20px', 'font-weight': 'bold'})]

# Define the update_progress function
def update_progress(progress_values, epochs):
    current_progress = len(progress_values) / epochs * 100

def data_is_not_loaded():
    # Add logic to check if data is loaded
    # For example, we can check if the 'loaded_data' variable is None
    return loaded_data is None

def clear_loaded_data():
    global loaded_data
    loaded_data = None

                                   
@app.callback(
    Output('lot-risk-score-card', 'style'),
    Output('recommendation-output', 'children'),
    [Input('train-predict-button', 'n_clicks')],
    [State('language-dropdown', 'value')]
)

def show_lot_risk_score_card(n_clicks, selected_language):
    global lot_score_percentage
    if n_clicks and n_clicks > 0:
        # Determine the recommendation based on the risk score
        if lot_score_percentage >= 20.0:
            recommendation_key = "High risk level: Short-Term Test recommended only."
        elif 10.0 <= lot_score_percentage < 20.0:
            recommendation_key = "Moderate risk level. Consider conducting further analysis."
        else:
            recommendation_key = "Low risk level : safe for vehicle installation."

        # Translate the recommendation message
        recommendation = translate_text(recommendation_key, selected_language)

        # If the execute button is clicked, show the card
        return {'width': '40%', 'margin-left': '20px', 'display': 'inline-block', 'border': '2px solid #000000',
                'border-radius': '5px', 'padding': '10px', 'margin-bottom': '20px', 'background-color': '#e0e0e0'}, \
                dcc.Markdown(f"{recommendation}")
    else:
        # Otherwise, hide the card
        return {'display': 'none'}, None



# Callback to train the model and predict
@app.callback(
    [Output('confusion-matrix', 'figure'),
     Output('conf-matrix-description', 'children'),
     Output('fig', 'figure'),
     Output('fig-table', 'figure'),
     Output('lot-risk-description','children'),
     Output('fig-metrics-table', 'figure'),
     Output('risk-scores-markdown', 'children'),
     Output('conf-matrix-store', 'data'),
     Output('fig-store', 'data'),
     Output('fig-table-store', 'data'),
     Output('fig-metrics-table-store', 'data'),
     Output('figure-train-store','data'),
     Output('fig-train', 'figure'),
     Output('figure-train-description', 'children')],
    [Input('train-predict-button', 'n_clicks'), 
    Input('language-dropdown', 'value')],
    prevent_initial_call=True
)


def train_and_predict(n_clicks, selected_language):
    
    global model, conf_matrix_fig, fig_metrics_table, fig,fig_table,conf_matrix_description, formatted_risk_score, loaded_data, \
           conf_matrix_store_data, figure_train, fig_store_data, fig_table_store_data, fig_metrics_table_store_data, \
           figure_train_store_data, figure_train_description, lot_risk_description, lot_score_percentage, \
           progress_values, conf_matrix_image_path, fig_image_path, fig_table_image_path, fig_metrics_table_image_path, \
           figure_train_image_path,conf_matrix, y_pred_binary

    if n_clicks is None or loaded_data is None:
        raise dash.exceptions.PreventUpdate
        # Assuming `loaded_data` is the new data from the server
    #loaded_data = preprocess_new_data(loaded_data)  # Preprocess the new data as needed

    # Split the data into features (X) and target variable (Y)
    X = loaded_data.drop('Risk_Level', axis=1)
    y = loaded_data['Risk_Level']

    # Use LabelEncoder to convert string labels to numerical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split the data into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Use StandardScaler for normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Specify parameters for XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
     }

    # Train the model
    epochs = 300
    
    # Initialize lists for progress tracking
    progress_values = []
    progress_interval = epochs // 100  # Update progress every 1% of epochs


    # Initialize lists outside the loop
    training_accuracy = []
    validation_accuracy = []
    training_loss = []
    validation_loss = []

    # Initialize an empty list for training results
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    results = {}
    
    model = xgb.train(params, dtrain, epochs, evals=evals, evals_result=results, verbose_eval=True)
    
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dtest)
    
    # Create a Plotly figure for the training history
    figure_train = go.Figure()
    
    # Extract training results from the results dictionary
    training_accuracy = results['train']['logloss']
    validation_accuracy = results['eval']['logloss']
    
    for epoch in range(epochs):
        # Update progress_values
        progress_values.append(epoch)

        if epoch % progress_interval == 0:
            update_progress(progress_values, epochs)
    
    # Add traces for training and validation accuracy
    figure_train.add_trace(go.Scatter(
        x=list(range(epochs)),
        y=training_accuracy,
        mode='lines',
        name='Training Accuracy'
    ))

    figure_train.add_trace(go.Scatter(
      x=list(range(epochs)),
      y=validation_accuracy,
       mode='lines',
    name='Validation Accuracy'
    ))
    
    # Set layout for the figure
    figure_train.update_layout(
        title=translate_text('XGBoost Training History', selected_language),
        xaxis={'title': translate_text('Training Time', selected_language)},
        yaxis={'title': translate_text('Accuracy/Loss', selected_language)},
        yaxis_tickformat='.2%'
    )

    figure_train_description = translate_text(
        '**Training Plot Overview:**\n\n'
        '- **Training History Plot:** Insights into model performance during training.\n'
        '- **Training Accuracy:** Model\'s learning progress from training data.\n'
        '- **Validation Accuracy:** Model\'s performance on a separate validation dataset.\n',
        selected_language
    )

    #confusion matrix
    # Make predictions on the test set
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_val_pred]
    y_val_binary = [1 if pred > 0.5 else 0 for pred in y_val_pred]

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_val_binary)
    
    # Calculate the predicted risk level distribution
    predicted_risk_counts = pd.Series(y_pred_binary).value_counts()
    risk_levels = ['Low Risk', 'High Risk']


    # Create a DataFrame with the predicted risk level distribution
    df_table = pd.DataFrame({
        'Risk Level': risk_levels,
        'No. of Cells': predicted_risk_counts,
        'Percentage': predicted_risk_counts / predicted_risk_counts.sum()
    })

    # Sort the DataFrame by the 'Percentage' column in descending order
    df_table = df_table.sort_values(by='Percentage', ascending=True)

    # Create a pie chart based on the sorted DataFrame
    fig = go.Figure(data=[go.Pie(labels=df_table['Risk Level'], values=df_table['No. of Cells'])])

    # Create a table with the sorted DataFrame
    fig_table = go.Figure(go.Table(
        header=dict(values=[
            translate_text('Risk Level', selected_language),
            translate_text('No. of Cells', selected_language),
            translate_text('Percentage', selected_language),
        ]),
        cells=dict(
            values=[
                df_table['Risk Level'],
                df_table['No. of Cells'],
                df_table['Percentage'],
            ],
            format=[
                None,
                None,
                '.2%',
            ]),
    ))
    
    # Update layout for the table
    fig_table.update_layout(
        height=300,
        showlegend=False,
        title=translate_text('Risk Level Table', selected_language),
    )

    # Update layout for the figure
    fig.update_layout(
        title=translate_text('Lot Risk Level Distribution', selected_language),
        showlegend=True,
    )

    # Translate lot risk description
    lot_risk_description = translate_text(
        f"**Lot Risk Level Distribution Overview:**\n\n"
        "- **Risk Level Pie Chart:** Visual representation of the distribution of risk levels across lots.\n"
        "- **Risk Level Table:** Detailed tabular data showing the number of cells and percentages of each risk level.\n\n"
        f"**Understanding the Results:**\n\n"
        "- **Low Risk:** Lots with a minimal risk of issues or defects.\n"
        "- **High Risk:** Lots with a significant risk of issues, requiring immediate attention.\n\n",
        selected_language
    )

    # Calculate the lot score based on the predicted values
    target_risk_level = 0 

    y_pred_binary_array = np.array(y_pred_binary)
    lot_score_count = (y_pred_binary_array == target_risk_level).sum()
    total_count = len(y_pred_binary_array)  # Total number of samples in the predictions

    # Calculate the percentage
    lot_score_percentage = round((lot_score_count / total_count) * 100, 2)
    formatted_risk_score = f"{lot_score_percentage:.2f}%"

    # Plot confusion matrix using Plotly Graph Objects
    conf_matrix_fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Low Risk', 'High Risk'],
        y=['Low Risk', 'High Risk'],
        colorscale='Viridis',
        zmin=0,
        showscale=True,
        hoverongaps=True,
        zmax=conf_matrix.max().max(),
        hoverinfo='z+text',
    ))

    # Add labels to the heatmap
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            conf_matrix_fig.add_annotation(
                x=['Low Risk', 'High Risk'][j],
                y=['Low Risk', 'High Risk'][i],
                text=str(conf_matrix[i][j]),
                showarrow=False,
                font=dict(color='black' if conf_matrix[i][j] > (conf_matrix.max().max() / 2) else 'white'),
            )

    # Add labels to the heatmap
    conf_matrix_fig.update_layout(
        title=translate_text('Confusion Matrix', selected_language),
        width=600,
        height=600,
        xaxis=dict(title=translate_text('Predicted Values', selected_language)),
        yaxis=dict(title=translate_text('Actual Data', selected_language)),
        annotations=[
            dict(x=0, y=0, text=f"TP: {conf_matrix[0, 0]}", showarrow=False, font=dict(size=14)),
            dict(x=1, y=0, text=f"FP: {conf_matrix[0, 1]}", showarrow=False, font=dict(size=14)),
            dict(x=0, y=1, text=f"FN: {conf_matrix[1, 0]}", showarrow=False, font=dict(size=14)),
            dict(x=1, y=1, text=f"TN: {conf_matrix[1, 1]}", showarrow=False, font=dict(size=14)),
        ]
    )

    # Reshape the confusion matrix for proper display
    conf_matrix_fig.update_layout(
        yaxis=dict(tickvals=list(range(len(conf_matrix)))),
    )
    
    # Calculate accuracy, precision, and recall
    accuracy = round((conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum(), 3)
    precision_high_risk = round(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]),3)
    recall_high_risk = round(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]),3)
    
    # Create a DataFrame with the metrics
    metrics_data = {
        'Metric': [
            translate_text('Accuracy', selected_language),
            translate_text('Manufacture Risk', selected_language),
            translate_text('Consumer Risk', selected_language)
        ],
        'Value': [accuracy, precision_high_risk, recall_high_risk]
    }

    metrics_df = pd.DataFrame(metrics_data)

    # Create a table with the metrics
    fig_metrics_table = go.Figure(go.Table(
        header=dict(values=[translate_text('Metric', selected_language), translate_text('Value', selected_language)]),
        cells=dict(values=[metrics_df['Metric'], metrics_df['Value']], format=[None, '.2%'])
    ))

    fig_metrics_table.update_layout(
        title_text=translate_text('Performance Metrics Table', selected_language),
        height=300,
        showlegend=False,
    )
    
    conf_matrix_translation = translate_text(
        '**Confusion Matrix Overview:**\n\n'
        '- **True Positives (TP):** {} cases correctly identified as High Risk.\n'
        '- **True Negatives (TN):** {} cases correctly identified as Low Risk.\n'
        '- **False Positives (FP):** {} cases incorrectly identified as High Risk.\n'
        '- **False Negatives (FN):** {} cases incorrectly identified as Low Risk.\n\n'
        '**Term Explanation:**\n'
        '- **True Positives (TP):** These are the cases where the model correctly identified items as High Risk, which is what we want.\n'
        '- **True Negatives (TN):** These are the cases where the model correctly identified items as Low Risk, which is also good.\n'
        '- **False Positives (FP):** These are the cases where the model incorrectly identified items as High Risk. Think of them as false alarms.\n'
        '- **False Negatives (FN):** These are the cases where the model incorrectly identified items as Low Risk. These are the misses we want to avoid.\n\n'
        '**Explanation:**\n'
        '- **Accuracy:** {} of predictions are correct.\n'
        '- **Manufacturing Risk:** {} of Risk predictions are correct.\n'
        '- **Consumer Risk:** {} of actual Risk cases are identified.\n',
        selected_language
    )

    # Format the translation with calculated values
    conf_matrix_description = conf_matrix_translation.format(
        conf_matrix[1, 1], conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0],
        f'{accuracy:.2%}', f'{precision_high_risk:.2%}', f'{recall_high_risk:.2%}'
    )

    # Convert figures to JSON strings and store them
    conf_matrix_store_data = pio.to_json(conf_matrix_fig) if conf_matrix_fig else None
    fig_store_data = pio.to_json(fig) if fig else None
    fig_table_store_data = pio.to_json(fig_table) if fig_table else None
    fig_metrics_table_store_data = pio.to_json(fig_metrics_table) if fig_metrics_table else None
    figure_train_store_data = pio.to_json(figure_train) if figure_train else None
    
    return (
        conf_matrix_fig,
        conf_matrix_description,
        fig,
        fig_table,
        lot_risk_description,
        fig_metrics_table,
        formatted_risk_score,
        conf_matrix_store_data,
        fig_store_data,
        fig_table_store_data,
        fig_metrics_table_store_data,
        figure_train_store_data,
        figure_train,
        figure_train_description,
    )

def generate_pdf_report(fig, conf_matrix_fig, fig_table, fig_metrics_table, figure_train,
                        conf_matrix_description, figure_train_description, lot_risk_description,
                        selected_language,file_path="report.pdf"):
    global conf_matrix, y_pred_binary
    try:
        print("Generating PDF report...")

        # Save the figures as base64-encoded images
        conf_matrix_image_base64 = fig_to_base64(conf_matrix_fig)
        fig_image_base64 = fig_to_base64(fig)
        fig_table_image_base64 = fig_to_base64(fig_table)
        fig_metrics_table_image_base64 = fig_to_base64(fig_metrics_table)
        figure_train_image_base64 = fig_to_base64(figure_train)
        
        # Calculate accuracy, precision, and recall
        accuracy = round((conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum(), 3)
        precision_high_risk = round(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]),3)
        recall_high_risk = round(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]),3)
        
        target_risk_level = 0 
        y_pred_binary_array = np.array(y_pred_binary)
        lot_score_count = (y_pred_binary_array == target_risk_level).sum()
        total_count = len(y_pred_binary_array)  # Total number of samples in the predictions

        # Calculate the percentage
        lot_score_percentage = round((lot_score_count / total_count) * 100, 2)
        formatted_risk_score = f"{lot_score_percentage:.2f}%"

        # Determine the recommendation based on the risk score
        if lot_score_percentage >= 20.0:
            recommendation_key = "High risk level: Short-Term Test recommended only."
        elif 10.0 <= lot_score_percentage < 20.0:
            recommendation_key = "Moderate risk level. Consider conducting further analysis."
        else:
            recommendation_key = "Low risk level: safe for vehicle installation."
            
        conf_matrix_description = f"<strong>{translate_text('Confusion Matrix Overview', selected_language)}:</strong>\n"

        # Add simple explanations for non-scientific users
        conf_matrix_description += (
            "<ul>"
            f"<li><strong>{translate_text('True Positives (TP)', selected_language)}:</strong> {conf_matrix[1, 1]} {translate_text('cases correctly identified as High Risk', selected_language)}.</li>"
            f"<li><strong>{translate_text('True Negatives (TN)', selected_language)}:</strong> {conf_matrix[0, 0]} {translate_text('cases correctly identified as Low Risk', selected_language)}.</li>"
            f"<li><strong>{translate_text('False Positives (FP)', selected_language)}:</strong> {conf_matrix[0, 1]} {translate_text('cases incorrectly identified as High Risk', selected_language)}.</li>"
            f"<li><strong>{translate_text('False Negatives (FN)', selected_language)}:</strong> {conf_matrix[1, 0]} {translate_text('cases incorrectly identified as Low Risk', selected_language)}.</li>"
            "</ul>"
            f"<strong>{translate_text('Term Explanation', selected_language)}:</strong>\n"
            "<ul>"
            f"<li><strong>{translate_text('True Positives (TP)', selected_language)}:</strong> {translate_text('These are the cases where the model correctly identified items as High Risk, which is what we want.', selected_language)}</li>"
            f"<li><strong>{translate_text('True Negatives (TN)', selected_language)}:</strong> {translate_text('These are the cases where the model correctly identified items as Low Risk, which is also good.', selected_language)}</li>"
            f"<li><strong>{translate_text('False Positives (FP)', selected_language)}:</strong> {translate_text('These are the cases where the model incorrectly identified items as High Risk. Think of them as false alarms.', selected_language)}</li>"
            f"<li><strong>{translate_text('False Negatives (FN)', selected_language)}:</strong> {translate_text('These are the cases where the model incorrectly identified items as Low Risk. These are the misses we want to avoid.', selected_language)}</li>"
            "</ul>"
        ).format(conf_matrix[1, 1], conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0])

        conf_matrix_description += f"<strong>{translate_text('Explanation', selected_language)}:</strong>\n"
        conf_matrix_description += (
            "<ul>"
            f"<li><strong>{translate_text('Accuracy', selected_language)}:</strong> {accuracy:.2%} {translate_text('of predictions are correct.', selected_language)}</li>"
            f"<li><strong>{translate_text('Manufacturing Risk', selected_language)}:</strong> {precision_high_risk:.2%} {translate_text('of Risk predictions are correct.', selected_language)}</li>"
            f"<li><strong>{translate_text('Consumer Risk', selected_language)}:</strong> {recall_high_risk:.2%} {translate_text('of actual Risk cases are identified.', selected_language)}</li>"
            "</ul>"
            f"<strong>{translate_text('Formulas', selected_language)}:</strong>\n"
            "<ul>"
            f"<li><strong>{translate_text('Accuracy', selected_language)}:</strong> {translate_text('Accuracy = (TP + TN) / (TP + TN + FP + FN)', selected_language)}</li>"
            f"<li><strong>{translate_text('Manufacturing Risk', selected_language)}:</strong> {translate_text('Manufacturing Risk = TP / (TP + FP)', selected_language)}</li>"
            f"<li><strong>{translate_text('Consumer Risk', selected_language)}:</strong> {translate_text('Consumer Risk = TP / (TP + FN)', selected_language)}</li>"
            "</ul>"
        ).format(accuracy, precision_high_risk, recall_high_risk)

        figure_train_description = """<strong>{}</strong>
            <ul>
                <li><strong>{}</strong> {}</li>
                <li><strong>{}</strong> {}</li>
                <li><strong>{}</strong> {}</li>
            </ul>""".format(
                translate_text('Training Plot Overview', selected_language),
                translate_text('Training History Plot', selected_language),
                translate_text('Insights into model performance during training.', selected_language),
                translate_text('Training Accuracy', selected_language),
                translate_text("Model's learning progress from training data.", selected_language),
                translate_text('Validation Accuracy', selected_language),
                translate_text("Model's performance on a separate validation dataset.", selected_language)
            )
        
        lot_risk_description = f"<strong>{translate_text('Lot Risk Level Distribution Overview', selected_language)}:</strong>\n"
        lot_risk_description += (
            "<ul>"
            f"<li><strong>{translate_text('Risk Level Pie Chart', selected_language)}:</strong> {translate_text('Visual representation of the distribution of risk levels across lots.', 'zh-CN')}</li>"
            f"<li><strong>{translate_text('Risk Level Table', selected_language)}:</strong> {translate_text('Detailed tabular data showing the number of cells and percentages of each risk level.', 'zh-CN')}</li>"
            "</ul>"
            f"<strong>{translate_text('Understanding the Results', selected_language)}:</strong>\n"
            "<ul>"
            f"<li><strong>{translate_text('Low Risk', 'zh-CN')}:</strong> {translate_text('Lots with a minimal risk of issues or defects.', 'zh-CN')}</li>"
            f"<li><strong>{translate_text('High Risk', 'zh-CN')}:</strong> {translate_text('Lots with a significant risk of issues, requiring immediate attention.', 'zh-CN')}</li>"
            "</ul>"
        )

        # Extracting CSS styles from dcc.Markdown
        figure_train_description_style = {
            'width': '90%',
            'margin-left': '23px'
        }

        conf_matrix_description_style = {
            'width': '90%',
            'margin-left': '23px'
        }

        lot_risk_description_style = {
            'width': '90%',
            'margin-left': '23px'
        }

        html_content = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: Calibri;
                        line-height: 1.5;
                        color: #333;
                        margin: 20px;
                        text-align: justify;
                    }}
                    h1, h2 {{
                        color: #000;
                    }}
                    .page-break {{
                        page-break-before: always;
                    }}
                    .figure-container {{
                        margin-bottom: 10px;
                    }}
                    .figure-number, .table-number {{
                        font-weight: bold;
                        font-size: 12px;
                        text-align: center;
                    }}
                    hr {{
                        margin-bottom: 20px;
                    }}
                    .lot-risk-card {{
                        width: 90%;
                        margin-left: 20px;
                        display: inline-block;
                        border: 2px solid #000000;
                        border-radius: 5px;
                        padding: 10px;
                        margin-bottom: 20px;
                        background-color: #e0e0e0;
                    }}
                    .center {{
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 70%;
                    }}
                </style>
                <title>{translate_text('Risk Analysis Summary', selected_language)}</title>
            </head>
            <body>
                <h1>{translate_text('Risk Analysis Summary Report', selected_language)}</h1>
                <hr>
                <p>{translate_text('Date', selected_language)}: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>{translate_text('Introduction', selected_language)}</h2>
                <p>
                    {translate_text('This report provides a brief overview and analysis of the risk associated with the dataset. The primary objectives of this analysis are to assess the lot risk level and provide recommendations based on the findings.', selected_language)}
                </p>

                <h2>{translate_text('Section 1. Training History', selected_language)}</h2>
                <div class="figure-container">
                    <img src="data:image/png;base64,{figure_train_image_base64}" alt="" class ="center">
                    <p class="figure-number">{translate_text('Fig 1. Training Plot', selected_language)}</p>
                </div>
                <p id='figure-train-description'>{figure_train_description}</p>
                <div class="page-break"></div>

               <h2>{translate_text('Section 2. Confusion Matrix Analysis', selected_language)}</h2>
                <div class="figure-container">
                    <img src="data:image/png;base64,{conf_matrix_image_base64}" alt="" class ="center">
                    <p class="figure-number">{translate_text('Fig 2. Confusion Matrix Heatmap', selected_language)}</p>
                </div>
                <div class="figure-container">
                    <img src="data:image/png;base64,{fig_metrics_table_image_base64}" alt="" class ="center" >
                    <p class="table-number">{translate_text('Table 1. Metrics Table', selected_language)}</p>
                </div>
                <p id='conf-matrix-description'>{conf_matrix_description}</p>

                <h2>{translate_text('Section 3. Lot Risk Level Distribution', selected_language)}</h2>
                <div class="figure-container">
                    <img src="data:image/png;base64,{fig_image_base64}" alt="" class ="center">
                    <p class="figure-number">{translate_text('Fig 3. Pie chart of Lot Risk Distribution', selected_language)}</p>
                </div>
                <div class="figure-container">
                    <img src="data:image/png;base64,{fig_table_image_base64}" alt="" class ="center">
                    <p class="table-number">{translate_text('Table 2. Risk Table', selected_language)}</p>
                </div>
                    <p id='lot-risk-description'>{lot_risk_description}</p>
                    
                <div class="lot-risk-card">
                    <h3>{translate_text('Lot Risk', selected_language)}</h3>
                    <div id='risk-scores-markdown' style='font-size: 20px; font-weight: bold; color: black;'>{formatted_risk_score}</div>
                    <hr>
                    <h5>{translate_text('Additional Insights and Recommended Action', selected_language)}</h5>
                    <div id='recommendation-output'>{translate_text(recommendation_key, selected_language)}</div>
                 </div>
            </body>
        </html>
        """
        
        # Convert HTML to PDF using wkhtmltopdf or another converter
        subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "-", file_path],
                       input=html_content.encode(), check=True)
        
        print(f"PDF generation successful. PDF saved to: {file_path}")

    except Exception as e:
        print(f"Error in generate_pdf_report: {e}")

    return file_path

def fig_to_base64(fig):
    # Convert a Plotly figure to a base64-encoded image
    return base64.b64encode(plotly.io.to_image(fig, format="png")).decode("utf-8")


button_clicks = 0

# Callback to generate and download the PDF report
@app.callback(
    Output("download", "data"),
    [Input("btn-download", "n_clicks")],
    [State('language-dropdown', 'value')],
    prevent_initial_call=True
)

def generate_pdf_report_callback(n_clicks, selected_language):
    global button_clicks

    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    button_clicks += 1

    # Check if the button has been clicked
    if button_clicks > 0:
        
        file_path = generate_pdf_report(
            fig,  
            conf_matrix_fig, fig_table, fig_metrics_table, figure_train,
            conf_matrix_description, figure_train_description, lot_risk_description,
            selected_language
        )

        return dcc.send_file(file_path)
    
    return dash.no_update


def read_html_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

@app.callback(
    [Output('fig-train', 'style'),
     Output('training-history-heading', 'children'),
     Output('confusion-matrix', 'style'),
     Output('confusion-matrix-heading', 'children'),
     Output('fig', 'style'),
     Output('risk-level-distribution-heading', 'children')],  
    [Input('train-predict-button', 'n_clicks'),
     Input('language-dropdown', 'value')],
    prevent_initial_call=True
)

def show_sections(n_clicks, selected_language):
    if n_clicks and n_clicks > 0:
        # If the execute button is clicked, show all sections

        # Translate the section headings based on the selected language
        translated_train_section = translate_text("Section 1. Training History", selected_language)
        translated_matrix_section = translate_text("Section 2. Confusion Matrix Analysis", selected_language)
        translated_risk_level_section = translate_text("Section 3. Lot Risk Level Distribution", selected_language)

        return (
            {'width': '70%', 'margin-left': '20px', 'display': 'inline-block'},
            translated_train_section,
            {'width': '40%', 'margin-left': '20px', 'margin-bottom': '20px'},
            translated_matrix_section,
            {'margin-left': '20px', 'display': 'block'},
            translated_risk_level_section,
        )
    else:
        # Otherwise, hide all sections
        return (
            {'display': 'none'},
            "",
            {'display': 'none'},
            "",
            {'display': 'none'},
            ""
        )

if __name__ == '__main__':
    app.run_server(mode = 'external', port = 8030, debug=True, host='0.0.0.0')
    
    # Open the app in a new browser window or tab
    webbrowser.open_new('http://0.0.0.0:8030')


# In[ ]:




