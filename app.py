# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd


model = None
df = None
app = Flask(__name__)


def select(df, attributes, ranges):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(attributes, list)
    assert isinstance(ranges, list)
    assert isinstance(ranges[0], list)

    selected_df = df.copy(deep=True)
    for i, attribute in enumerate(attributes):
        selected_df = selected_df[selected_df[attribute].isin(ranges[i])]
    return selected_df


def load_model():
    global model
    # model variable refers to the global variable
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)


def get_pd_df():
    return pd.read_csv('./data/data_to_show.csv')


@app.route('/register', methods=['GET', 'POST'])
def register():
    pass


@app.route('/')
def home_endpoint():
    global df
    df = get_pd_df()
    neighbourhood_group_set = set(df['neighbourhood_group'])
    neighbourhood_set = set(df['neighbourhood'])

    return render_template('index.html', tables=[df.head().to_html(classes='data', header='true')])


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data_json = request.get_json()  # Get data posted as a json
        data = data_json['data']
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)