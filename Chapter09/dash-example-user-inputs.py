# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
Chapter 9. Hands-On Predictive Analytics with Python
Building a basic interactive app
"""
## imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import os

## Importing the dataset
DATA_DIR = '../data'
FILE_NAME = 'diamonds.csv'
data_path = os.path.join(DATA_DIR, FILE_NAME)
diamonds = pd.read_csv(data_path)
diamonds = diamonds.sample(n=2000)


app = dash.Dash(__name__)

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

numerical_features = ['price','carat','depth','table','x','y','z']
options_dropdown = [{'label':x.upper(), 'value':x} for x in numerical_features]

dd_x_var = dcc.Dropdown(
        id='x-var',
        options = options_dropdown,
        value = 'carat'
        )

div_x_var =  html.Div(
        children=[html.H4('Variable for x axis: '), dd_x_var],
        className="six columns"
        )
        

dd_y_var = dcc.Dropdown(
        id='y-var',
        options = options_dropdown,
        value = 'price'
        ) 

div_y_var =  html.Div(
        children=[html.H4('Variable for y axis: '), dd_y_var],
        className="six columns"
        )

app.layout = html.Div(children=[
        html.H1('Adding interactive controls'),
        html.H2('Interactive scatter plot example'),
        html.Div(
                children=[div_x_var, div_y_var],
                className="row"
                ),  
        dcc.Graph(id='scatter')
        ])


@app.callback(
        Output(component_id='scatter', component_property='figure'),
        [Input(component_id='x-var', component_property='value'), Input(component_id='y-var', component_property='value')])
def scatter_plot(x_col, y_col):
    trace = go.Scatter(
            x = diamonds[x_col],
            y = diamonds[y_col],
            mode = 'markers'
            )
    
    layout = go.Layout(
            title = 'Scatter plot',
            xaxis = dict(title = x_col.upper()),
            yaxis = dict(title = y_col.upper())
            )
    
    output_plot = go.Figure(
            data = [trace],
            layout = layout
            )
    
    return output_plot
 
      
if __name__ == '__main__':
    app.run_server(debug=True)