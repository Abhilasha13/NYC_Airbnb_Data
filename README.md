# NYC_Airbnb_data
Exploratory Data Analysis and Data Preprocessing was performed on New York City Airbnb dataset

## INTRODUCTION: 
Online booking systems have been popular since a long time. The shift to the booking system has been a boon as it has opened several options on the fingertips. One such stay booking system that has been known since a long time is AirBnb.  

Airbnb is an online marketplace for arranging or offering lodging, primarily homestays, or tourism experiences since 2008. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Amongst the various places in the world popular for renting out Airbnb’s, NYC stands out to be a contender. NYC is the most populous city in the United States and also one of the most popular tourism and business place in the world. These two factors combined, gives us a successful data analysis project to work upon. 

Through the process of data analysis and predictive modelling, I have tried to solve the following problem statements:
• To identify the major attributes that attract a greater number of bookings to the different Airbnbs.
• To create a geospatial economic model to determine the pricing of Airbnbs.
• To understand the concept of dynamic pricing and affordability of different listings.

## DATA COLLECTION AND DATASET DESCRIPTION: 
This dataset was taken from [insideairbnb](http://insideairbnb.com/get-the-data.html) which is publicly available. This dataset contains listing activity and the various metrics used to identify how popular the property is and the factors affecting it. 

The dataset contains 49k+ rows and 76 attributes, containing 36 categorical variable and 40 continuous variables are present.

## DATA PREPROCESSING: 
The data pre-processing was performed into three major segments. They have been listed below: 

1. Data Cleaning – Count and percentage of total records having Null values for each attribute was calculated. The attributes with more than 50% null values were removed. Columns like ‘host_response_time’ , ‘host_response_rate’ were removed. 

2. Replacing Null Values – There were 6 attributes which had information about review ratings. Each of these columns had few records with Null values as ratings. These null values were replaced with the median value of the existing review rating corresponding to that column. 

3. Data Binning – Continuous and categorical attributes were categorized into lesser number of bins. The number of bins range from 2-7 depending on the attribute data. Columns such as “host_list_count”, “avail365” have been simply binned. Certain columns required some pre-processing before binning could take place. Columns such as “amenities” and “host_verification” were in the form of list of values. The count has been taken from the list and then binned accordingly. The column “bathroom_text” has been split into 2 different attributes based on the space delimiter. Those 2 attributes have been binned separately.

4. Data Transformation: After initial data modeling attempts. The need for creating new attributes was realized. One of the most important drivers of the dependent variables was Amenities. A unique formula was designed to quantify the amenities though a weighted average approach and generated a new attribute named “Amenities Score”. The idea behind the weighted average approach was to inverse rank the amenities with most occurrence. Here is the formula used:
                                             
                                             Amenities Score: Σ(𝑎𝑚𝑒𝑛𝑖𝑡𝑖𝑒𝑠(𝑟𝑎𝑛𝑘)∗𝑎𝑚𝑒𝑛𝑖𝑡𝑖𝑒𝑠(𝑜𝑐𝑐𝑢𝑟𝑎𝑛𝑐𝑒))/Σ𝑎𝑚𝑒𝑛𝑖𝑡𝑖𝑒𝑠(𝑟𝑎𝑛𝑘)

## EXPLORATORY DATA ANALYSIS: 
In order to get a better insight of the AirBnb business, exploratory data analysis was performed and the individual attributes were explored to understand the feature relations.

The analysis shows top 25 most frequently occurred amenities in the dataset and a word cloud was shown for better visualization. Through this analysis it could be inferred that the demand of any listing is directly proportional to the different facilities available in that listing. Moreover, it could also means that to attract a potential customer a new host or any existing host must have these facilities.

It was also analyzed that the room type of any listing has a good correlation with the price of any listing. The plot represents the Airbnb room type vs price of any listing. It can be observed that if a listing is in a hotel room, the price is really high when compared to the entire home or shared room.

The count of room type available in different neighborhood locations is also shown. It can be visualized that the number of hotel rooms are very less in different neighborhood regions. This helped to conclude that most of the customers prefer to book an entire home or a private apartment as compared to a hotel room. And this could also be one of the reasons why the price of an entire apartment or a private room is within a moderate range.

One of the problem statements is to create a geospatial economic model to determine the price of an Airbnb and so, the distribution of Airbnb listings and the price variation across the neighborhood areas was analyzed. It can be inferred that the Airbnb’s are less in demand at Staten Island and it might also be possible that there are a lesser number of visitors or tourists who visit here for the purpose of business or sight-seeing. Hence, this may also be one of the reasons why the price of a listing in Staten Island is low. Also, it can be observed that Brooklyn and Queens have a greater distribution of Listings and so, the price of properties in these places is moderate.

## FEATURE SELECTION
In order to find the relationship between two or more variables and to understand how the different features are related on our dependent variable (i.e. ‘price’), correlation was performed. The pre-processed dataset consists approximately 28 continuous variables and so, Pearson’s correlation was used to select the most important features which was also used for our predictive analysis.

It was observed that the variables like Accommodates, bedrooms, beds, and availability_365 have a positive correlation with price. Also, longitude and number of reviews have a negative correlation with price. There was no correlation between ‘id’ and ‘price’.

For feature selection, Random Forest ensemble algorithm was used. This algorithm was run to perform feature selection on the categorical variables. This helped to extract only the required or the most important features and eliminate the less significant features. The variables that contributed most to our dependent variables were room_type, binned_accomodate, neighborhood_group_cleansed, binned_beds, binned_bedrooms, binned_host_list_count and binned_hyear.

## DATA PREDICTION
To understand the concept of dynamic pricing and affordability of properties better, predictive analysis was performed. This helped to decide what could be the best predictive model for price. For predictive modelling, Classification algorithms (Supervised Learning) and Regression was used.

### Classification Algorithms:
To execute the classification algorithms, the preprocessed dataset was first divided into training and testing sets. 70% data was chosen randomly to build the training set and the rest 30% was used to create the testing set. The training set was then used to train the model for prediction and then using the testing set the accuracy of prediction was measured. After dividing the data into training and testing sets, the team used Decision tree (C5.0) and Naïve Bayes to predict the price variation. Initially the two algorithms were executed using all the attributes in the dataset.

• Decision Tree – Due to less computational time, and maximum sensitivity, C5.0 Decision tree algorithm was used for prediction. It was observed that the accuracy of prediction using all the attributes was 79.5%. However, after executing the algorithm using the top 8 most important algorithm, it was observed that the accuracy increased to 82.83%.

• Naïve Bayes – Naïve Bayes algorithm was also used for prediction due to its simplicity, good speed and high scalability. The accuracy achieved through Naïve Bayes was approximately 72.61%. A significant increase in the prediction accuracy was observed when the Naïve Bayes algorithm was run using only the top 8 most important features obtained as a result feature selection and correlation.

### Regression Algorithms:
The most important attributes were taken into a separate dataframe and the null values were checked. The only column having null values is reviews_per_month. These null values were replaced with zeroes. Then the categorical columns were identified and one hot encoding was performed on them. This is done as machine learning models cannot operate on label data directly. Thus the categorical variables are converted to numeric variables.

• Linear Regression – A multiple linear regression model was run to determine the price of a property. Linear regression gives the baseline model upon which the model is improved with further techniques. Linear regression was chosen, since it is fairly simple and requires the least computational time. The R2 value of 0.094062014755396, a mean absolute error of 78.74113382109259 and a mean squared error of 316.3492954986619 was achieved.

• Lasso Regression – Lasso regression was used as it has the ability to nullify parameters that do not improve the model. It is a type of linear regression that uses a concept called shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean. This is done by adding a little bit of bias and reducing the variance greatly. An R2 value of 0.09406155592682797, a mean absolute error of 78.72277662683376 and a mean squared error of 316.3471968793032 was obtained. This was identified as an insignificant improvement from the previous regression model.

• Random Forest Regression – This regression technique was selected as this is an ensemble bagging algorithm and generates an internal unbiased estimate of generalization error. Ensemble method combines the predictions from multiple machine learning algorithms so as to make more accurate predictions. In bagging, random sampling is done with replacement. It allowed the team to better understand the bias and variance that was introduced in the earlier regression technique. It involves random sampling of small subset of data from the dataset. Random forest regression technique is constructed by a multitude of decision trees at training time. The trees in random forests run in parallel and have no interactions with each other while building the model. Initially the model runs into a problem of overfitting because complete and full trees were used. This overfitted model gives an R2 value of 0.8964396441209034, a mean absolute error of 69.94758268288254 and a mean squared error of 240.360953371544.

To tackle the issue of overfitting, hyperparameter tuning was performed through a trial method. This involves pruning the trees by tuning the parameters so that a better result was obtained. As a result, R2 value of 0.5148539953972002, a mean absolute error of 68.67836498963754 and a mean squared error of 258.22717606117556 was obtained.

## CONCLUSION
1. The results from the implemented models, and the exploratory data analysis showed that Airbnb is widespread across different localities in New York and the price depends mostly on the location and the type of room or property that is being booked. 

2. The impact of amenities on the pricing of a property was also explored. Fire-safety is an important factor while determining price of a property. Smoke alarms are important in determining the reliability of a property. 

3. Weighted amenities score was calculated and concluded that it does not have a positive correlation with the price, thus amenities are secondary to location and room type. 

4. The Airbnb market is distributed all around with different pricing in different locations. Both classification and regression algorithms were implemented and the results would also help a buyer chose an Airbnb property in a better manner.

## CHALLENGES 
Couple of challenges and blockers were encountered while trying to work with the dataset. They have been listed below: 

1. Since the initial dataset, consisted of 76 attributes, the team faced the challenge to intuitively and collectively decide on the columns that would not be of any use in either analysis or prediction. Columns such as ‘host_url’, ’review_description’ have been removed from the dataset.

2. Prior to pre-processing, the bathroom attribute had values such as “Half-bath”, “Shared-bath”, “2 Private Baths”. The data was split into two columns and certain assumptions were made to clean up the data.

## USAGE INSTRUCTION

Use Jupyter notebook to run the give code.

## REFERENCES

[Data Exploration on NYC Airbnb . (n.d.).](https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb)

[Airbnb Analysis, Visualization and Prediction. (n.d.)](https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction)
