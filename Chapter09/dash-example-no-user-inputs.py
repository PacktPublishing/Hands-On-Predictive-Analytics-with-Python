# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
Chapter 9. Hands-On Predictive Analytics with Python
Building a basic static app
"""
## imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import os

## Importing the dataset
DATA_DIR = '../data'
FILE_NAME = 'diamonds.csv'
data_path = os.path.join(DATA_DIR, FILE_NAME)
diamonds = pd.read_csv(data_path)

## Creating the app
app = dash.Dash(__name__)

# Creating a Plotly figure
trace = go.Histogram(
        x = diamonds['price']
        )

layout = go.Layout(
        title = 'Diamond Prices',
        xaxis = dict(title='Price'),
        yaxis = dict(title='Count')
        )

figure = go.Figure(
        data = [trace],
        layout = layout
        )

app.layout = html.Div([
        html.H1('My first Dash App'),
        html.H2('Histogram of diamond prices'),
        html.P('This is some normal text, we can use it to describe something about the application.'),          
        dcc.Graph(id='my-histogram', figure=figure)
        ])
        
      
if __name__ == '__main__':
    app.run_server(debug=True) 
        