# Serve model as a flask application

import pickle
from flask import Flask, request, render_template
import pandas as pd
from predict import data_transform, predict, parse_request

from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import Category20c
from bokeh.transform import linear_cmap
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import ColorBar, NumeralTickFormatter
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.palettes import RdBu

from viz_FilterbyText.pipeline_new_1 import viz_key_df
from viz_FilterbyText.pipeline_new import visualize_count, visualize_price, donut

model = None
df = None
application = Flask(__name__)


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


def get_ng_dict(df):
    ng_dict = {}
    for _, row in df.iterrows():
        ng = row['neighbourhood_group']
        if ng not in ng_dict.keys():
            ng_dict[ng] = set()
        ng_dict[ng].add(row['neighbourhood'])

    for key, val in ng_dict.items():
        ng_dict[key] = list(val)
    return ng_dict


def load_model():
    global model
    # model variable refers to the global variable
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)


def get_pd_df(path):
    return pd.read_csv(path)


def parse_price_range(priceRangeStrList):
    priceRangeList = []
    for priceRangeStr in priceRangeStrList:
        loStr, hiStr = priceRangeStr.split('-')
        if hiStr == '':
            priceRangeList.append(list(range(int(loStr), 10000)))
        else:
            priceRangeList.append(list(range(int(loStr), int(hiStr))))
    return priceRangeList


def select_from_request(result, notfound = False):
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
        
        
        
    
    neighbourhoodList = result.getlist('neighbourhood')
    if neighbourhoodList != []:
        attributes.append('neighbourhood')
        ranges.append(neighbourhoodList)
    if notfound: 
        return select(df, attributes, ranges)
    # priceRangeList = result.getlist('priceRange')
    # if priceRangeList != []:
    #     attributes.append('price')
    #     ranges.append(parse_price_range(priceRangeList))

    priceRange = '-'
    min_price = result.get('minPrice')
    max_price = result.get('maxPrice')
    if min_price or max_price:
        if min_price:
            priceRange = min_price + priceRange
        else:
            priceRange = '0' + priceRange
        if max_price:
            priceRange = priceRange + max_price

        attributes.append('price')
        ranges.append(parse_price_range([priceRange]))


    min_nights = result.get('minNight')
    if min_nights:
        attributes.append('minimum_nights')
        ranges.append(int(min_nights))

    min_reviews = result.get('minReview')
    if min_reviews:
        attributes.append('number_of_reviews')
        ranges.append(int(min_reviews))
        
    return select(df, attributes, ranges)


def plot_bokeh_map_new(df_new):
    # df_new=pd.read_csv('./data/final_dataframe.csv', index_col=0)

    room_list = ['bedroom','bedrooms','bed','beds','bdrs','bdr','room','rooms']
    fs = df_new.copy()
    # fs = fs[fs["neighbourhood"].isin(["Kips Bay"])]

    return viz_key_df(room_list, fs)


def plot_bokeh_map(): ###NEED TO ADD PASS-IN PARAM:pd.DataFrame (filtered df)!!!####

    ###ADDED#### 
    #   IF above param is filled, then remove the ADDED SECTION
    dataframe = pd.read_csv('./data/final_dataframe.csv', index_col=0)
    ###ADDED####

    room_list = ['bedroom', 'bedrooms', 'bed',
                 'beds', 'bdrs', 'bdr', 'room', 'rooms',
                 'apt','apartment','studio','loft','townhouse',
                 'bath','baths']

    def sort_keys(ls, df): 
        all_room_df = df[df.title_split.str.contains('|'.join(ls))]
        return all_room_df

    df = sort_keys(room_list, dataframe)


    chosentile = get_provider(Vendors.OSM)
    palette = RdBu[7]
    source = ColumnDataSource(data=df)
    # Define color mapper - which column will define the colour of the data points
    color_mapper = linear_cmap(field_name = 'price', palette = palette, low = df['price'].min(), high = df['price'].max())

    # Set tooltips - these appear when we hover over a data point in our map, very nifty and very useful
    tooltips = [("Price","@price"), ("Region","@neighbourhood"), ("Keywords","@title_split"), ("Number of Reviews", "@number_of_reviews")]
    # Create figure
    p = figure(title = 'Airbnb Listings Selected', 
               x_axis_type = "mercator", 
               y_axis_type = "mercator", 
               x_axis_label = 'Longitude', 
               y_axis_label = 'Latitude', 
               tooltips = tooltips,
               plot_width=1350, plot_height=800)
    # Add map tile
    p.add_tile(chosentile)
    # Add points using mercator coordinates
    sz = 2
    if (len(df) < 70) or (df["neighbourhood"].nunique() == 1): 
        sz = 12
    p.circle(x = 'mercator_x', y = 'mercator_y', color = color_mapper, source = source, size = sz, fill_alpha = 0.9)
    #Defines color bar
    color_bar = ColorBar(color_mapper=color_mapper['transform'], 
                         formatter = NumeralTickFormatter(format='0.0[0000]'), 
                         label_standoff = 13, width=8, location=(0,0), title = "price")

    p.add_layout(color_bar, 'right')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    p.title.align = "right"

    # output_notebook()
    # show(p)
    # print(len(components(p)))
    script1, div1 = components(p)
    cdn_js = CDN.js_files[0]
    # cdn_css = CDN.css_files[0]
    return script1, div1, cdn_js    


@application.route('/', methods=['POST', 'GET'])
def home_endpoint():
    col_to_show = ['name', 'host_name', 'room_type', 'neighbourhood_group',
                   'neighbourhood',
                   'minimum_nights', 'number_of_reviews', "price"]
    global df
    df = get_pd_df('./data/final_dataframe.csv')
    df_selected = df.copy(deep = True).drop_duplicates(col_to_show).reset_index()
    roomTypeSet = set(sorted(set(df['room_type'])))
    neighbourhoodGroupSet = set(sorted(set(df['neighbourhood_group'])))
    neighbourhoodSet = set(sorted(set(df['neighbourhood'])))

    anchor = "top"

    
    ng_dict = get_ng_dict(df)
    msg_pred = str(len(df_selected)) + "records found based on given inputs, Average Price is: $" + str(round(df_selected["price"].mean(), 1)) + ", Median Price is: $" + str(round(df_selected["price"].median(), 1)) + ", displaying top 20 cheapest offerings: "
    # select data according to the submitted form
    
    if request.method == 'POST':
        anchor = "finder"
        df_selected = select_from_request(request.form).drop_duplicates(col_to_show).reset_index()
        
        if len(df_selected) == 0:
            encoded_input = data_transform(df, request.form)
            price_predicted = predict('model.pkl', encoded_input)
            
            
            #print(price_predicted)
            msg_pred = "We have no available record that match the searching input, but our model recommands a reasonable price based on the market trend" 
            msg_pred = msg_pred + " for the given inputs is: " + "$" + str(price_predicted) + ". " 
            msg_pred = msg_pred + " \n And we can provide the closest matches in the region: "
            request_pd = parse_request(df, request.form).to_frame().T
            request_pd.insert(0, 'predicted_price', price_predicted)
            #df_selected = request_pd
        elif len(df_selected) < 20: 
            encoded_input = data_transform(df, request.form)
            price_predicted = predict('model.pkl', encoded_input)
            msg_pred = "Less than 20 records found based on the inputs, which may not be representative of the market. Based on our model, a resonable price recommended for the given inputs is: " 
            msg_pred = msg_pred + "$" + str(price_predicted) + ". " 
            
            msg_pred = msg_pred + " \n And the available listings found based on the inputs are: "
        
        
        
        
        else: 
            
            msg_pred = str(len(df_selected)) + "records found based on given inputs, Average Price is: $" + str(round(df_selected["price"].mean(), 1)) + ", Median Price is: $" + str(round(df_selected["price"].median(), 1)) + ", displaying top 20 cheapest offerings: "
            
            
            

    
    
    
    if len(df_selected) == 0: 
        df = get_pd_df('./data/final_dataframe.csv')  # No result found, use the closest matches instead 
        df_selected = select_from_request(request.form, notfound = True).drop_duplicates(col_to_show).reset_index()
             
    for i in ng_dict.keys(): 
        ng_dict[i] = list(sorted(ng_dict[i]))
    
    script1, div1, cdn_js = plot_bokeh_map_new(df_selected)
    script1_count, div1_count, cdn_js_count = visualize_count(df_selected)
    script1_price, div1_price, cdn_js_price = visualize_price(df_selected)
    
    img = donut(df_selected)

    #df_selected = df_selected.drop_duplicates().reset_index()
    if len(df_selected) >= 20: 
        df_selected = df_selected.head(20)
    #df_selected = pd.concat([df_selected.drop("price", axis = 1), df_selected["price"]], axis = 1)     
    return render_template('index.html', anchor=anchor, request_form=request.form,
                           selected_RT = request.form.getlist('roomType'),
                           selected_NG = request.form.getlist('neighbourhoodGroup'),
                           selected_NEI = request.form.get('neighbourhood'),
                           
                           
                           tables = [df_selected[col_to_show].rename({"name": "Title", "host_name": "Host", "neighbourhood_group": "Region", "neighbourhood": "Neighbourhood", "room_type": "Room Type", "minimum_nights": "Minimum Nights", "number_of_reviews": "Number of Reviews", "price": "Price Per Day"}, axis = 1).to_html(classes='data', header='true')],
                           roomTypeSet = sorted(roomTypeSet),
                           neighbourhoodGroupSet = sorted(neighbourhoodGroupSet),
                           neighbourhoodSet = neighbourhoodSet, ng_dict = ng_dict,
                           script1=script1, div1=div1, cdn_js=cdn_js, msg_pred=msg_pred,
                           script1_count=script1_count, div1_count=div1_count, cdn_js_count=cdn_js_count,
                           script1_price=script1_price, div1_price=div1_price, cdn_js_price=cdn_js_price,
                           img=img)


# @application.route('/predict', methods=['POST'])
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
    
    application.run(host='0.0.0.0', port=5000, debug = True, use_reloader = False)
