# Serve model as a flask application

from crypt import methods
import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
from sympy import true


model = None
df = None
app = Flask(__name__)


def select(df, attributes, ranges):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(attributes, list)
    assert isinstance(ranges, list)
    # assert isinstance(ranges[0], list)

    selected_df = df.copy(deep=True)
    for i, attribute in enumerate(attributes):
        if isinstance(ranges[i], list):
            if isinstance(ranges[i][0], list):
                for r in ranges[i]:
                    selected_df = selected_df[selected_df[attribute].isin(r)]
            else:
                selected_df = selected_df[selected_df[attribute].isin(ranges[i])]
        else:
            selected_df = selected_df[selected_df[attribute] > ranges[i]]

    return selected_df


def load_model():
    global model
    # model variable refers to the global variable
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)


def get_pd_df():
    return pd.read_csv('./data/data_to_show.csv')


def parse_price_range(priceRangeStrList):
    priceRangeList = []
    for priceRangeStr in priceRangeStrList:
        loStr, hiStr = priceRangeStr.split('-')
        if hiStr == '':
            priceRangeList.append(list(range(int(loStr), 10000)))
        else:
            priceRangeList.append(list(range(int(loStr), int(hiStr))))
    return priceRangeList


@app.route('/', methods=['POST', 'GET'])
def home_endpoint():
    global df
    df = get_pd_df()
    df_selected = df.copy(deep=true)
    roomTypeSet = set(df['room_type'])
    neighbourhoodGroupSet = set(df['neighbourhood_group'])
    neighbourhoodSet = set(df['neighbourhood'])

    if request.method == 'POST':
        result = request.form
        attributes = []
        ranges = []

        roomTypeList = result.getlist('roomType')
        if roomTypeList != []:
            attributes.append('room_type')
            ranges.append(roomTypeList)
        
        neighbourhoodGroupList = result.getlist('neighbourhoodGroup')
        if neighbourhoodGroupList != []:
            attributes.append('neighbourhood_group')
            ranges.append(neighbourhoodGroupList)
        
        priceRangeList = result.getlist('priceRange')
        if priceRangeList != []:
            attributes.append('price')
            ranges.append(parse_price_range(priceRangeList))

        min_nights = result.get('minNight')
        if min_nights:
            attributes.append('minimum_nights')
            ranges.append(int(min_nights))

        min_reviews = result.get('minReview')
        if min_reviews:
            attributes.append('number_of_reviews')
            ranges.append(int(min_reviews))
        df_selected = select(df, attributes, ranges)

    return render_template('index.html',
                           tables=[df_selected.head().to_html(
                               classes='data', header='true')],
                           roomTypeSet=roomTypeSet,
                           neighbourhoodGroupSet=neighbourhoodGroupSet,
                           neighbourhoodSet=neighbourhoodSet)



# @app.route('/predict', methods=['POST'])
# def get_prediction():
#     # Works only for a single sample
#     if request.method == 'POST':
#         data_json = request.get_json()  # Get data posted as a json
#         data = data_json['data']
#         # converts shape from (4,) to (1, 4)
#         data = np.array(data)[np.newaxis, :]
#         # runs globally loaded model on the data
#         prediction = model.predict(data)
#     return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)
