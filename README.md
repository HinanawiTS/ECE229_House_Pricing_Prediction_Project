# New York City Airbnb: Find Your Dream House

The housing rental market in New York City is gradually recovering after a rough time during the pandemic. The economy in NYC also revives with the reopening of restaurants, the growth of tourism, and the rapid rising of housing rentals. It is crucial for travelers, long-term tenants, and landlords to understand the housing rental trend in NYC and locate their desired offers, either to stay for a few nights or to optimize the pricing strategy for their own properties. In this project, we provide local hosts and prospective travellers with an interactive and easy to use system that helps them to better understand the rental housing market in New York City.

This system is deployed on AWS servers (edit: the server is no longer available) and it includes sophisticated modules for data pipeline, analytical models, results, visualization, and interactive on-the-fly analysis.

## User Story
As a travel/renter who wants to get the best housing offerings in the NYC Airbnb rental market, I would like to receive an overview of the rental market in the region I interested in, so I can make informed purchase decisions. To be more specific, I expect the system to provide available offerings filtered by my inputs (locations, price range, etc), and automatically generate analysis and visualizations that are easy to understand.

As a homeowner in NYC who wants to lease my properties on Airbnb, I also want an overview of the rental market in my region so I can decide a competitive pricing strategy for my offering. Moreover, I would like to know the number of offerings that are similar to my property, and the average price of those offerings. I would like those information to be visualized on a map so it's easier to digest. And if there is no similar offering exist, I would like to have a price recommended for my property based on the market, so it can serve as a baseline for my pricing strategy. 

## Visit the Website 
The AWS account hosting the website has been terminated and the AWS server is no longer available. To run the website locally, directly run the application.py at the root directory. 

Demo: 

https://user-images.githubusercontent.com/35885069/188051339-1ef9b4bb-459e-4417-9e7f-624c16c0944c.mp4


http://ec2-52-27-61-221.us-west-2.compute.amazonaws.com:5000/ (No longer available)
The detailed explanations for the use of the application are included in the "How to Use" section of the website.  

## Documentation
Documentation for python functions are hosted at: https://hinanawits.github.io/ECE229-Documentation/

## Test Coverage Reports 
![cs_reports](coverage_reports/coverage_reports.png)
Generated by: pytest --cov=application --cov=viz_FilterbyText.pipeline_new_1 --cov=viz_FilterbyText.pipeline_new --cov-report term-missing

Some functions for rendering webpages in application.py were not tested because they require flask.request object which is inputed by web requests. Also, a large portion of the project was implemented in HTML/CSS/Javascrpt, and we only used the results of the EDA and modeling, the process of those parts were not included in our application in order to save computational resources. 

## Repository Structure 
    .
    
    ├── assets                          # Assets fpr website 
    
    ├── coverage_reports                # Coverage reports generated by the test cases  
    
    ├── data  
        ├── AB_NYC_2019.csv             # Original dataset 
        ├── data_feature.csv            # Added feature
        ├── data_preprocessed.csv       # Preprocessed dataset 
        ├── data_to_show.csv            # Data for visualization 
        ├── final_dataframe.csv         # Data for analysis 
    
    ├── docs                            # Contains all packages necessary to build the sphinx documentation 
    
    ├── pred 
        ├── model.pkl                   # Trained weights of the Decision Tree Regressor 
        ├── predict.py                  # Source for making predictions 
    
    ├── static                          # Contains all static resources for the website 
    
    ├── template
        ├── index.html                  # Static part of the website 
        ├── actual_app.html             # Application Website 
    
    ├── viz_FilterbyText 
        ├── pipeline_new.py             # Visualization pipeline for hexbin maps and donut chart 
        ├── pipeline_new_1.py           # Visualization pipeline for bokeh map 
        
    ├── Application.py                  # A flask backend, including all necessary backend functions for the website 
    ├── Dockerfile                      # Docker file for website deployment 
    ├── docker-compose.yml              # Docker yml for website deployment 
    ├── test_cases.py                   # Test cases for all Python functions we implemented 
    ├── .gitignore              
    ├── requirements.txt
    └── README.md










