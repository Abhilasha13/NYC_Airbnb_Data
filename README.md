# NYC_Airbnb_data
Exploratory Data Analysis and Data Preprocessing was performed on New York City Airbnb dataset

## INTRODUCTION: 
Online booking systems have been popular since a long time. The shift to the booking system has been a boon as it has opened several options on the fingertips. One such stay booking system that has been known since a long time is AirBnb.  

Airbnb is an online marketplace for arranging or offering lodging, primarily homestays, or tourism experiences since 2008. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Amongst the various places in the world popular for renting out Airbnb’s, NYC stands out to be a contender. NYC is the most populous city in the United States and also one of the most popular tourism and business place in the world. These two factors combined, gives us a successful data analysis project to work upon. 

## RELEVANT PRIOR WORK: 
While learning more about the dataset, I came across several Kaggle notebooks which used similar kind of dataset for performing their analysis, to help get an idea of the various insights that can be gained from the dataset and various perspectives. 

One of the Kaggle notebooks, Airbnb NYC Price Prediction, helped us get an understanding about the dependency relationship of price attribute with other attributes, while the other, NYC Airbnb, aided in the visualizations through which other attributes could be shown. 

## DATA COLLECTION AND DATASET DESCRIPTION: 

This dataset was taken from [insideairbnb](http://insideairbnb.com/get-the-data.html) which is publicly available. This dataset contains listing activity and the various metrics used to identify how popular the property is and the factors affecting it. 

The dataset contains 49k+ rows and 76 attributes, containing 36 categorical variable and 40 continuous variables are present.

## DATA PREPROCESSING: 
The data pre-processing was performed into three major segments. They have been listed below: 

1. Data Cleaning – Count and percentage of total records having Null values for each attribute was calculated. The attributes with more than 50% null values were removed. Columns like ‘host_response_time’ , ‘host_response_rate’ were removed. 

2. Replacing Null Values – There were 6 attributes which had information about review ratings. Each of these columns had few records with Null values as ratings. These null values were replaced with the median value of the existing review rating corresponding to that column. 

3. Data Binning – Continuous and categorical attributes were categorized into lesser number of bins. The number of bins range from 2-7 depending on the attribute data. Columns such as “host_list_count”, “avail365” have been simply binned. Certain columns required some pre-processing before binning could take place. Columns such as “amenities” and “host_verification” were in the form of list of values. The count has been taken from the list and then binned accordingly. The column “bathroom_text” has been split into 2 different attributes based on the space delimiter. Those 2 attributes have been binned separately.

## EXPLORATORY DATA ANALYSIS: 
In order to get a better insight of the AirBnb business, exploratory data analysis was performed and the individual attributes were explored to understand the feature relations.

The analysis represents top 25 most frequently occurred amenities in the dataset and a word cloud for better visualization. Through this analysis it could be inferred that the demand of any listing is directly proportional to the different facilities available in that listing. Moreover, it also means that to attract a potential customer a new host or any existing host must have these facilities. 

It was also analyzed that the room type of any listing has a good correlation with the price of any listing. The above plot represents the Airbnb room type vs price of any listing. Also, the second figure represents the count of room type available in different neighborhood locations.  

## CHALLENGES 
Couple of challenges and blockers were encountered while trying to work with the dataset. They have been listed below: 

1. Since the initial dataset, consisted of 76 attributes, the team faced the challenge to intuitively and collectively decide on the columns that would not be of any use in either analysis or prediction. Columns such as ‘host_url’, ’review_description’ have been removed from the dataset.

2. Prior to pre-processing, the bathroom attribute had values such as “Half-bath”, “Shared-bath”, “2 Private Baths”. The data was split into two columns and certain assumptions were made to clean up the data.

## Next Steps
The next step includes building predictive models and using algorithms like SVM, Decision Tree, Regression, etc.

## Usage Instruction

Use Jupyter notebook to run the give code.

## References

1. [Data Exploration on NYC Airbnb . (n.d.).] (https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb)
2. [Airbnb Analysis, Visualization and Prediction. (n.d.)] (https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction)
