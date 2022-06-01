import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder


def parse_request(df, request_form):
    inputed = {}
    inputed['neighbourhood'] = request_form.get("neighbourhood")
    inputed['neighbourhood_group'] = request_form['neighbourhoodGroup']
    inputed['room_type'] = request_form['roomType']
    inputed['minimum_nights'] = (int) (request_form['minNight'])
    inputed['availability_365'] = (int) (df['availability_365'].mean())

    return pd.Series(inputed)


def data_transform(df, request_form):
    encoded = df
    encoded = encoded.drop(["host_name", "latitude", "longitude", "number_of_reviews", "reviews_per_month",
                        "id", "last_review", "host_id", "calculated_host_listings_count"], axis=1)
    encoded = encoded[encoded["price"] < 1000]
    encoded = encoded.drop(["name", "price"], axis=1)
    
    inputed = parse_request(df, request_form)
    encoded = encoded.append(inputed, ignore_index=True)

    encoded = pd.get_dummies(
        encoded, prefix = ["group", "ne", "room_type"], drop_first = True)
    encoded_input = encoded.iloc[-1]

    return encoded_input


def predict(model_path, encoded_input):
    lr2 = pickle.load(open(model_path, 'rb'))
    return lr2.predict([encoded_input])[0]