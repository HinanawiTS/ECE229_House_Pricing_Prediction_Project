from pred import predict
from application import *
from viz_FilterbyText.pipeline_new_1 import *
from viz_FilterbyText.pipeline_new import *
import pytest
import pandas as pd

class TestCases:
    ##### Chang #####
    def test_predict(self):
        pass

    def test_get_ng_dict(self):
        pass

    def test_load_model(self):
        pass

    def test_get_pd_df(self):
        pass
    ##### Chang #####


    ##### Kexin #####
    

    def test_plot_bokeh_map_new(self):
        ''' 
        Tests plot_bokeh_map_new() fucntion in application.py. The plot_bokeh_map_new() function 
        is used to filter room_lists that help plot bokeh figure.
        The plot_bokeh_map_new function input should be a structured dataframe and returns a function 
        called viz_key_df with passed in params: room_list and df.
        '''
        rm_ls=['bedroom', 'bedrooms', 'bed',
                 'beds', 'bdrs', 'bdr', 'room', 'rooms',
                 'apt', 'apartment', 'studio', 'loft', 'townhouse',
                 'bath', 'baths']   
        csv_path='./viz_FilterbyText/final_dataframe.csv'
        full_df=pd.read_csv(csv_path, index_col=0)
        random_df=full_df.sample(n=10) # randomly select 10 rows from full_df
        assert type(random_df) == pd.DataFrame
        output1,output2,output3 = plot_bokeh_map_new(random_df)
        assert isinstance(output1, str) 
        assert isinstance(output2, str)
        assert isinstance(output3, str)


    def test_plot_bokeh_smalldf(self):
        '''
        Tests plot_bokeh_smalldf() fucntion in viz_FilterbyText.pipeline_new_1.py. The plot_bokeh_smalldf() function 
        is used to create an auto-generated bokeh figure from the final_dataframe.csv.
        The plot_bokeh_smalldf function input should be a structured dataframe and returns a bokeh plot based on it.
        '''
        csv_path='./viz_FilterbyText/final_dataframe.csv'
        full_df=pd.read_csv(csv_path, index_col=0)
        random_df=full_df.sample(n=10) # randomly select 10 rows from full_df
        col_name_check=['id','name','host_id','host_name','neighbourhood_group','neighbourhood',
                        'latitude','longitude','room_type','price','minimum_nights','number_of_reviews','coordinates',
                        'mercator','mercator_x','mercator_y','title_split']
            
        assert all(i in col_name_check for i in random_df.columns.tolist())  == True
        # self.assertEqual(str(type(p)),"<class 'bokeh.plotting.figure.Figure'>") # comment out as return object is not a bokeh fig
        script1, div1, cdn_js=plot_bokeh_smalldf(random_df)
        assert isinstance(cdn_js, str) and cdn_js=='https://cdn.bokeh.org/bokeh/release/bokeh-2.3.2.min.js'
        assert isinstance(script1, str)
        assert isinstance(div1, str)
    ##### Kexin #####


    ##### bittan #####
    def test_parse_price_range(self):
        pass

    def test_select_from_request(self):
        pass

    def test_sort_keys(self):
        pass

    def test_viz_key_df(self):
        pass
    ##### bittan #####