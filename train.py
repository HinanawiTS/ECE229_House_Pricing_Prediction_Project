
# import os
# from os.path import dirname
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import ExtraTreesClassifier

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# #from sklearn.metrics import accuracy_score
# #from sklearn.metrics import confusion_matrix


# #from sklearn.model_selection import KFold


# d = os.getcwd()
# parent_d = dirname(d)
# filepath = parent_d + '/data/data_preprocessed.csv'
# df = pd.read_csv(filepath)
# df.head()

 
# # ## feature selection


# #  Log transformation on price. make less skewed
# def log_model(df):
#     df_log = df.copy()
#     df_log['price_log'] = np.log(df_log.price+1)
#     df_log.drop(columns=['id','host_id', 'last_review_ts','price'], axis = 1, inplace = True)
#     return df_log


# df_log = log_model(df)
# df_log


# corr = df_log.corr(method='kendall')
# plt.figure(figsize=(15,8))
# sns.heatmap(corr, annot=True)
# df_log.columns


# # Train test splitting with desired parameters.
# def get_train_test_split(df, test_size = 0.2, random_state = 1, scale=True):
#         '''
#         @param:
#             df: dataframe to be split
#             test_size: Test data size as the fraction of whole data size.
#             random_state: Random state to be used in splitting.
#         @return:
#             Train features, train labels, test features, test labels are returned in separate Pandas Dataframes in a tuple size of 4.
#         '''
#         assert isinstance(test_size, (float, int))
#         assert isinstance(random_state, (int))
        
#         model_x, model_y = df.iloc[:,:-1], df.iloc[:,-1] # df.loc[:, df.columns != 'price']   df.loc[:, df.columns == 'price']
        
#         if scale==True:
#             scaler = StandardScaler()
#             model_x = scaler.fit_transform(model_x)
#         else:
#             model_x = model_x.copy()
            
#         X_train, X_test, y_train, y_test = train_test_split(model_x, model_y , test_size=test_size, random_state=random_state)
        
#         return X_train, X_test, y_train, y_test


# def feature_importance(df,scale=True):
#     '''
#         @param:
#             scale: True by default, scales the features.
#         @return:
#             dataframe with final features.
#     '''
            
#     # Get Train and Test data
#     X_train, X_test, y_train, y_test = get_train_test_split(df, test_size = 0.2, random_state = 1)
    
#     lab_enc = preprocessing.LabelEncoder()

#     feature_model = ExtraTreesClassifier(n_estimators=50)
#     feature_model.fit(X_train,lab_enc.fit_transform(y_train))

#     plt.figure(figsize=(7,7))
#     feat_importances = pd.Series(feature_model.feature_importances_, index=df.iloc[:,:-1].columns)
#     feat_importances.nlargest(10).plot(kind='barh')
#     plt.show()


# feature_importance(df_log)


# def drop_column(df,save=True):
#     df_new = df.copy()
#     df_new.drop(columns=['id','host_id', 'last_review_ts','neighbourhood_group', 'room_type'], axis = 1, inplace = True)
#     index = ['neighbourhood', 'latitude', 'longitude',
#              'minimum_nights', 'number_of_reviews','reviews_per_month', 
#              'calculated_host_listings_count', 'availability_365','price',]
#     df_new=df_new[index]
    
#     if save==True:
#         df_new.to_csv(parent_d + '/data/data_feature.csv', index=False)
        
#     return df_new


# df_feature = drop_column(df,save=True)
# df_feature

 
# # ## model selection


# def make_predictions(df, model = 'lr', cv=5, tune = False, save = False):
#     '''
#      @param:
#         df: dataframe to be used
#         model: Name of the model 
#         tune: tune the hyper parameters or not
#         save: save the result, to /data folder.
#     @return:
#         predictions on the test data with the original test features.
#     '''

#     # Get Train and Test data
#     X_train, X_test, y_train, y_test = get_train_test_split(df, test_size = 0.2, random_state = 1, scale=True)
    
#     # Training and Prediction with Different Models
#     #  Linear Regression
#     if model == 'lr':
#         if tune == True:
#             clf_lr = LinearRegression()
#             parameters = {'copy_X':[True, False], 'fit_intercept':[True,False], 'normalize':[True,False]}
            
#             grid_search_lr = GridSearchCV(estimator=clf_lr,  
#                          param_grid=parameters,
#                          scoring='neg_mean_squared_error',
#                          cv=cv,
#                          n_jobs=-1)
            
#             grid_search_lr.fit(X_train, y_train)
#             best_parameters_lr = grid_search_lr.best_params_  
#             best_score_lr = grid_search_lr.best_score_ 
#         else:
#             best_parameters_lr =  {'copy_X':True, 'fit_intercept': True, 'normalize': True}
        
#         clf = LinearRegression(**best_parameters_lr)
    
#     # Ridge Regression
#     if model == 'rr':
#         if tune == True:
#             clf_rr = Ridge()
#             parameters = {'alpha' :np.array([1,0.1,0.01,0.001,0.0001,0]),'normalize':[True,False]}           
            
#             grid_search_rr = GridSearchCV(estimator=clf_rr,  
#                          param_grid=parameters,
#                          scoring='neg_mean_squared_error',
#                          cv=cv,
#                          n_jobs=-1)
            
#             grid_search_rr.fit(X_train, y_train)
#             best_parameters_rr = grid_search_rr.best_params_  
#             best_score_rr = grid_search_rr.best_score_ 
#         else:
#             best_parameters_rr =  {'alpha':0.01, 'normalize': True}
            
#         clf = Ridge(**best_parameters_rr)
    
#     # Lasso Regression
#     if model=='l':
#         if tune == True:
#             clf_l =  Lasso()
#             parameters = {'alpha' :np.array([1,0.1,0.01,0.001,0.0001,0]),'normalize':[True,False]}           
            
#             grid_search_l = GridSearchCV(estimator=clf_l,  
#                          param_grid=parameters,
#                          scoring='neg_mean_squared_error',
#                          cv=cv,
#                          n_jobs=-1)
            
#             grid_search_l.fit(X_train, y_train)
#             best_parameters_l = grid_search_l.best_params_  
#             best_score_l = grid_search_l.best_score_ 
#         else:
#             best_parameters_l =  {'alpha':0.001, 'normalize' :False}
            
#         clf = Lasso(**best_parameters_l)
        
#     # ElasticNet Regression 
#     if model=='er':
#         if tune == True:
#             clf_er =  ElasticNet()
#             parameters = {'alpha' :np.array([1,0.1,0.01,0.001,0.0001,0]),'normalize':[True,False]}           
            
#             grid_search_er = GridSearchCV(estimator=clf_er,  
#                          param_grid=parameters,
#                          scoring='neg_mean_squared_error',
#                          cv=cv,
#                          n_jobs=-1)
            
#             grid_search_er.fit(X_train, y_train)
#             best_parameters_er = grid_search_er.best_params_  
#             best_score_er = grid_search_er.best_score_ 
#         else:
#             best_parameters_er =  {'alpha': 0.01, 'normalize':False}
            
#         clf =  ElasticNet(**best_parameters_er)
        
                          
#     # Random Forest Regressor
#     if model=='rfr':
#         if tune == True:
            
#             clf_rfr = RandomForestRegressor()
#             max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
#             max_depth.append(None)
#             min_samples_split = [2,5,10]
#             min_samples_leaf = [1, 2, 4]
#             bootstrap = [True, False]
#             parameters = {'n_estimators' :[int(x) for x in np.linspace(start = 200, stop=2000,num=10)],
#                           'max_features':['auto', 'sqrt'],
#                           'max_depth' :max_depth,
#                           'min_samples_split':[2,5,10],
#                           'min_samples_leaf':[1, 2, 4],
#                           'bootstrap': [True, False]}
#             grid_search_rfr = GridSearchCV(estimator=clf_rfr,  
#                          param_grid=parameters,
#                          scoring='neg_mean_squared_error',
#                          cv=cv,
#                          n_jobs=-1)
            
#             grid_search_rfr.fit(X_train, y_train)
#             best_parameters_rfr = grid_search_rfr.best_params_  
#             best_score_rfr = grid_search_rfr.best_score_ 
#         else:
#             best_parameters_rfr =  {'n_estimators': 100, 'max_features':'sqrt','max_depth':None,
#                                     'min_samples_split':2, 'min_samples_leaf':1, 'bootstrap': True}
            
#         clf =  RandomForestRegressor(**best_parameters_rfr)
        
#     # XGBoost Regressor
#     if model == 'xgb':
#         if tune == True:
#             clf_xgb = XGBRegressor()
#             parameters = { 'learning_rate': [0.01, 0.05, 0.1, 0.5],
#                            'max_depth': [3,5,7,9],
#                            'min_child_weight':[1,3,5]}

#             grid_search_xgb= GridSearchCV(estimator=clf_xgb,  
#                          param_grid=parameters,
#                          scoring='neg_mean_squared_error',
#                          cv=cv,
#                          n_jobs=-1)
            
#             grid_search_xgb.fit(X_train, y_train)
#             best_parameters_xgb = grid_search_xgb.best_params_  
#             best_score_xgb = grid_search_xgb.best_score_ 

#         else:
#             best_parameters_xgb  =  {'learning_rate':0.1, 'max_depth':3, 'min_child_weight':3}
        
#         clf = XGBRegressor(**best_parameters_xgb)                                         
                          

#     # model fit, predict
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     #probs = clf.predict_proba(X_test)
#     #accuracy = accuracy_score(y_test, pred)
#     #conf = confusion_matrix(y_test, pred)
    
#     x = X_test.copy()
#     y = y_test.copy()
#     index = ['neighbourhood', 'latitude', 'longitude',
#              'minimum_nights', 'number_of_reviews','reviews_per_month', 
#              'calculated_host_listings_count', 'availability_365']
#     df_pred = pd.DataFrame(x, columns = index)
#     #df_pred['price'] = y
#     df_pred['pred'] = pred
#     #df_pred['prob_1'] = probs[:, 1]
                
#     if save == True:
#         df_pred.to_csv(parent_d + '\data\predictions.csv', index = False)
    
#     print(model)
#     print('MAE: %f'% mean_absolute_error(y_test, pred))
#     print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, pred)))   
#     print('R2 %f' % r2_score(y_test, pred))
                
#     return df_pred            


# #Linear Regression
# make_predictions(df_feature, model = 'lr', cv=5, tune = True, save = False)


# ##Ridge Regression
# make_predictions(df_feature, model = 'rr', cv=5, tune = True, save = False)


# ##Lasso Regression
# make_predictions(df_feature, model = 'l', cv=5, tune = True, save = False)


# ##ElasticNet Regression 
# make_predictions(df_feature, model = 'er', cv=5, tune = True, save = False)


# # XGBoost Regressor
# make_predictions(df_feature, model = 'xgb', cv=5, tune = True, save = False)


# # Random Forest Regressor
# make_predictions(df_feature, model = 'rfr', cv=5, tune = True, save = False)








