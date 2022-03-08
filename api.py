import json
import flask
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = flask.Flask(__name__)
CORS(app)

def read(json_file):
    return json.load(open(json_file))


def unwrap(tensor):
    return tensor.numpy()[0][0]


def format(input_dict):
    formatted = {name: np.array([value])
                 for name, value in input_dict.items()}
    return formatted


@app.route('/apartment', methods=['POST'])
def get_apartment_prediction():
    model = tf.keras.models.load_model('models/apartment')
    data = format(flask.request.get_json())
    return {"Prediction" : int(unwrap(model(data)))}
