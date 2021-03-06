import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
import json
from skimage import io
from skimage.transform import resize
import neural_network as nn
import numpy as np
import pandas as pd


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.COSMO],assets_ignore="base")
server = app.server

canvas_width = "100%"
bg_filename = "https://l-kershaw.github.io/images/border.jpg"

n = nn.init_trained_network("./trained_network/init_data.csv","./trained_network/weights.csv")

app.head = [
	html.Meta(content="width=device-width, initial-scale=1",name="viewport")
	]


banner = dbc.Row(
	className="banner",
	children=[
		dbc.Col(
			[
				html.A(
					id="back",
					children=[html.Img(src=app.get_asset_url("back.png"),alt="back")],
					href="https://l-kershaw.github.io/coding/2020-10-character-recognition",
					style={"float":"left","margin-left":"10px","margin-right":"10px"}
				),
				html.H2(
					"Character Recognition",
					style={"margin-left":"100px"}
				)
			],
			width={"size": 12, "order": "first"},
			md={"size": 7, "order": "first"},
		),
		dbc.Col(
			[
				html.A(
						id="gh-link",
						children=["View on GitHub"],
						href="https://github.com/l-kershaw/character-recognition",
						style={"color": "white", "border": "solid 1px white"},
						className="vertical-center"
				),
				html.Img(
					src=app.get_asset_url("GitHub-Mark-Light-64px.png"),
					className="vertical-center"
				),
			],
			width={"size": 12},
			md={"size": 4},
			id="gh",
			
		)
	],
	justify="between"
)

cards = dbc.Row(
			[
			dbc.Col(width=0,md=1),
			dbc.Col(
				dbc.Card(
					[
						DashCanvas(
							id='draw-canvas',
							tool='pencil',
							#width=canvas_width,
							lineWidth=20,
							lineColor='black',
							filename=bg_filename,
							hide_buttons=['pencil','line','zoom','pan','rectangle','select'],#,'undo','redo'],
							goButtonTitle='Process')
					],
					color="light"
				),
				width=12,
				md=5,
				lg=4
			),
			dbc.Col(
				dbc.Card(
					[
						html.Img(id='my-image',width="100%")
					],
				),
				width=12,
				md=5,
				lg=4
			),
			dbc.Col(width=1,lg=0),
			dbc.Col(width=1,lg=0),
			dbc.Col(
				dbc.Card(
					[html.Div(id='digit-output',children=[])],
				),
				width=10,
				lg=2
			),
			dbc.Col(width=1)#,md=1)
			]
		)

app.layout = html.Div(
	className="",
	children=[
		banner,
		cards
	]
)


# @app.callback(Output('display-value', 'children'),
							# [Input('dropdown', 'value')])
# def display_value(value):
		# return 'You have selected "{}"'.format(value)

# app.clientside_callback(
    # """
    # function(largeValue1) {
        # return window.innerWidth.toString()
    # }
    # """,
    # Output('width-div', 'value'),
    # [Input('test-input', 'value')]
# )



@app.callback([Output('my-image', 'src'),Output('digit-output', 'children')],
							[Input('draw-canvas', 'json_data')])
def update_data(string):
	if string:
		imshape = io.imread(bg_filename, as_gray=True).shape
		full_data = parse_jsonstring(string, (imshape[0],imshape[1]))
	else:
		raise PreventUpdate
	full_data = [[int(x)*(0.998)+0.001 for x in row] for row in full_data]
	# print(data)
	full_data = np.array(full_data)
	full_data = resize(full_data,imshape)
	
	data = full_data[1:29,1:29]
	flat_data = data.flatten()
	# print(data)
	output = n.query(flat_data)
	output = output.flatten()
	# Rescale outputs
	out_sum = float(sum(output))
	output = [(i,output[i]/out_sum) for i in range(len(output))]
	# Sort outputs in decreasing likelihood order
	output.sort(reverse=True,key=lambda x : x[1])
	
	# Format guesses as strings
	guesses = [str(x[0])+": "+ "{:.2f}%".format(x[1]*100) for x in output]
	# Wrap guesses in <p> elements
	guesses = [html.P(x) for x in guesses]
	# Only return top 5 guesses
	guesses = guesses[:5]
	return array_to_data_url((255-255 * full_data).astype(np.uint8)), guesses

if __name__ == '__main__':
	app.run_server(debug=True)


