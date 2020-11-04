from flask import Flask, render_template, request, send_from_directory
from skimage import io
from skimage.transform import resize
import numpy as np
# import keras.models
import neural_network as nn
import re
import base64

import sys
import os
sys.path.append(os.path.abspath("./trained_network"))
from load import *

# UPLOAD_FOLDER = '/images/'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'css'}

app = Flask(__name__)#,
            # static_url_path='',
            # static_folder='static',
            # template_folder='templates')

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global model, graph
# model, graph = init()

n = nn.init_trained_network("./trained_network/init_data.csv","./trained_network/weights.csv")

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
	# get data from drawing canvas and save as image
	parseImage(request.get_data())

	# read parsed image back in 8-bit, black and white mode (L)
	x = io.imread('images/processed.png', as_gray=True)
	x = 1-x
	x = resize(x,(28,28))

	# reshape image data for use in neural network
	x = x.reshape(1,784)
	output = n.query(x)
	print(output)
	output = output.flatten()
	# Rescale outputs
	out_sum = float(sum(output))
	output = [(i,output[i]/out_sum) for i in range(len(output))]
	# Sort outputs in decreasing likelihood order
	output.sort(reverse=True,key=lambda x : x[1])

	# Format guesses as strings
	guesses = [str(x[0])+": "+ "{:.2f}%".format(x[1]*100) for x in output]
	# # Wrap guesses in <p> elements
	# guesses = [html.P(x) for x in guesses]
	# Only return top 5 guesses
	guesses = guesses[:5]
	return str(guesses)
	# with graph.as_default():
	#     out = model.predict(x)
	#     print(out)
	#     print(np.argmax(out, axis=1))
	#     response = np.array_str(np.argmax(out, axis=1))
	#     return response

@app.route('/prepare/', methods=['GET','POST'])
def prepare():
	# get data from drawing canvas and save as image
	parseImage(request.get_data())

	x = io.imread('images/processed.png', as_gray=True)
	print(x)
	# x = 1-x
	x = resize(x,(28,28))
	io.imsave('images/processed.png',x)
	return 'images/processed.png'

@app.route('/images/<path:filename>')
def download_file(filename):
	return send_from_directory('images',
							   filename, as_attachment=True)


# @app.route('/static/<path:path>')
# def send_css(path):
# 	print('accessing css')
# 	return send_from_directory('static', path)

def parseImage(imgData):
	# parse canvas bytes and save as output.png
	imgstr = re.search(b'base64,(.*)', imgData).group(1)
	with open('images/processed.png','wb') as output:
		output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
	app.debug = True
	port = int(os.environ.get("PORT", 5000))
	app.run()#host='0.0.0.0', port=port)
