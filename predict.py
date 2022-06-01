import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder



def parse_request(df, request_form):
    inputed = {}
    inputed['neighbourhood'] = request_form.get("neighbourhood")
    inputed['neighbourhood_group'] = request_form['neighbourhoodGroup']
    inputed['room_type'] = request_form['roomType']
    if len(request_form['minNight']) > 0: 
  
        inputed['minimum_nights'] = int(request_form['minNight'])
    else:
        
        inputed['minimum_nights'] = 7
    inputed['availability_365'] = 112

    return pd.Series(inputed)


def data_transform(df, request_form):
    encoded = df
    encoded = encoded[["price", "neighbourhood_group", "neighbourhood", "room_type", "minimum_nights"]] 
    
    encoded["availability_365"] = 112 
    encoded = encoded[encoded["price"] < 1000]
    encoded = encoded.drop(["price"], axis=1)
    
    inputed = parse_request(df, request_form)
    encoded = encoded.append(inputed, ignore_index = True)

    #print(encoded)
    encoded = pd.get_dummies(
        encoded, prefix = ["group", "ne", "room_type"], drop_first = True)
    encoded_input = encoded.iloc[-1]
    #print(encoded)
    return encoded_input


def predict(model_path, encoded_input):
    lr2 = pickle.load(open(model_path, 'rb'))
    
    return lr2.predict([encoded_input])[0]