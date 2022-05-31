import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder


def parse_request(request_form):
    # inputed = pd.Series({"neighbourhood_group": "Manhattan", "neighbourhood": "Arden Heights",
    #                      "room_type": "Private room", "minimum_nights": 0, "availability_365": 0})
    inputed = {}
    inputed['neighbourhood_group'] = request_form['neighbourhoodGroup']
    inputed['neighbourhood'] = request_form.get("neighbourhood")
    inputed['room_type'] = request_form['roomType']
    inputed['minimum_nights'] = request_form['minNight']
    inputed['availability_365'] = 90   # hard code for future modification

    return pd.Series(inputed)


def data_transform(df, request_form):
    dataset = df.drop(["id", "host_id", "host_name", "last_review"], axis = 1)
    encoded = dataset.copy()
    for i in ['neighbourhood_group', 'neighbourhood', 'room_type']:
        en = OneHotEncoder()
        transformed = en.fit_transform(dataset[[i]]).toarray()
        columns = en.categories_[0]
        for j in range(len(columns)): 
            columns[j] = str(i) + "_" + str(columns[j])
        ohe_df = pd.DataFrame(transformed, columns = columns)
        encoded = pd.concat([encoded, ohe_df], axis = 1).drop([i], axis = 1)

    encoded["reviews_per_month"] = encoded["reviews_per_month"].fillna(0)
    def lux(x): 
        if pd.isna(x["name"]): 
            return 0
        elif "lux" in x["name"]: 
            return 1
        else: 
            return 0
    encoded["lux"] = encoded.apply(lux, axis = 1)

    encoded = encoded.drop(["name"], axis = 1)
    encoded = df.drop(["host_name", "latitude", "longitude", "number_of_reviews", "reviews_per_month",
                       "id", "last_review", "host_id", "calculated_host_listings_count"], axis=1)
    encoded = encoded[encoded["price"] < 1000]
    encoded = encoded.drop(["name", "price"], axis=1)
    inputed = parse_request(request_form)
    encoded = encoded.append(inputed, ignore_index=True)

    encoded = pd.get_dummies(
        encoded, prefix=["group", "ne", "room_type"], drop_first=True)
    encoded_input = encoded.iloc[-1]
    return encoded_input


def predict(model_path, encoded_input):
    lr2 = pickle.load(open(model_path, 'rb'))
    return lr2.predict([encoded_input])[0]