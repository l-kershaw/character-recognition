import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_canvas import DashCanvas

app = dash.Dash(__name__)
server = app.server

canvas_width = 300
bg_filename = "https://l-kershaw.github.io/images/border.jpg"

app.layout = html.Div([
    html.H2('Hello World'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='display-value'),
		DashCanvas(	id='canvas',
								tool='pencil',
								width=canvas_width,
								height=1000,
								lineColor='black',
								filename=bg_filename,
								hide_buttons=['pencil','line','zoom','pan','rectangle','select'],#,'undo','redo'],
								goButtonTitle='Process')
])
@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)
if __name__ == '__main__':
    app.run_server(debug=True)