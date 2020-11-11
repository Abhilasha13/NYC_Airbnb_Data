#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib
from wordcloud import WordCloud

#Importing Airbnb dataset from excel
data = pd.read_excel("./Airbnb_V1_clean.xlsx")
data.head(5)

# Dataset description
data.describe().T
percentage_missing_data = pd.DataFrame([data.isnull().sum(), data.isnull().sum() * 100.0/data.shape[0]]).T
percentage_missing_data.columns = ['No. of Missing Data', 'Percentage of Missing data']
percentage_missing_data

# Processing 'host_since' columns
data['host_since']= pd.to_datetime(data['host_since'])
data['host_since_year']=pd.DatetimeIndex(data['host_since']).year

# Binning 'host_since' columns
cut_labels_hyear = ['2008-2010', '2010-2012', '2012-2014', '2014-2016','2016-2018','2018-2020']
cut_bins_hyear = [2008.0, 2010.0, 2012.0, 2014.0, 2016.0, 2018.0, 2020.0]
data['binned_hyear'] = pd.cut(data['host_since_year'], bins=cut_bins_hyear, labels=cut_labels_hyear, include_lowest= True)

# Deleting rows with null values
data = data.dropna(axis=0, subset=['host_since'])

# Counting number of elements in 'host_verifications' columns
host_verifications_count = []
for i in data["host_verifications"].items():
    host_verifications_count.append(i[1].count(',')+1)

data["host_verifications_count"] = host_verifications_count

# Binning 'host_verifications' columns
labels_host_verifications_count = ['0 to 4','5 to 8','9 above']
bins_host_verifications_count = [0,4,8,np.inf]
data['binned_host_verifications_count'] = pd.cut(data['host_verifications_count'], bins_host_verifications_count, labels=labels_host_verifications_count, include_lowest= True)
data['binned_host_verifications_count'].value_counts()

# Binning 'accommodates' columns
bins_accomodate = [0, 1, 2, 3, 4, 5, 6, np.inf]
labels_accomodate =[1,2,3,4,5,6,'7 above']
data['binned_accomodate'] = pd.cut(data['accommodates'], bins_accomodate, labels=labels_accomodate)

data['binned_accomodate'].value_counts()

# Replacing null values with 0
data['bedrooms'] = data['bedrooms'].replace(np.nan, 0)
data['beds'] = data['beds'].replace(np.nan, 0)

# Binning 'bedrooms' columns
bins_bedrooms = [0, 1, 2, 3, 4, 5,np.inf]
labels_bedrooms =[1,2,3,4,5,'5 above']
data['binned_bedrooms'] = pd.cut(data['bedrooms'], bins_bedrooms, labels=labels_bedrooms, include_lowest= True)

# Binning 'beds' columns
bins_beds = [0,1, 2, 3, 4, np.inf]
labels_beds =[1,2,3,4,'5 above']
data['binned_beds'] = pd.cut(data['beds'], bins_beds, labels=labels_beds, include_lowest= True)

# Binning 'price' columns
labels_price = ['<=100', '>100']
bins_price = [0, 100.00,np.inf]
data['binned_price'] = pd.cut(data['price'], bins=bins_price, labels=labels_price, include_lowest= True)

data['price'].median()
data[['price','binned_price']].head()
data['binned_price'].value_counts()

# Counting number of elements in 'amenities' columns
amenities_count = []
for i in data["amenities"].items():
    amenities_count.append(i[1].count(',')+1)

data["amenities_count"] = amenities_count

# Binning 'amenities' columns
labels_amenities_count = ['0 to 10','11 to 20','21 to 30','31 to 40','41 to 50','51 to 60','60 above']
bins_amenities_count = [0,10,20,30,40,50,60,np.inf]
data['binned_amenities_count'] = pd.cut(data['amenities_count'], bins_amenities_count, labels=labels_amenities_count, include_lowest= True)

# Splitting column 'bathrooms_text' into 'bathrooms_count' and 'bathrooms_type'
data[['bathrooms_count','bathrooms_type']] = pd.DataFrame([x.split(' ') for x in data['bathrooms_text'].tolist()])

data["bathrooms_count"].unique()
data["bathrooms_type"].unique()

# Replacing nan and spaces with 0:
data['bathrooms_count'] = data['bathrooms_count'].replace(np.nan, 0)
data['bathrooms_count'] = data['bathrooms_count'].replace('', 0)
data['bathrooms_type'] = data['bathrooms_type'].replace('', 0)

data["bathrooms_count"] = data.bathrooms_count.astype(float)

# Binning 'bathrooms_type' columns
value_dict_bathrooms_type  = {'bath':'Private','private':'Private','Private':'Private','baths':'Shared','shared':'Shared','Shared':'Shared',0:'Shared'}
data["binned_bathrooms_type"] = data["bathrooms_type"].replace(value_dict_bathrooms_type)

# Binning 'bathrooms_count' columns
labels_bathrooms_count = ['0 to 1', '1.5 to 2', '2.5 to 3', '3.5 to 4','4 above']
bins_bathrooms_count = [0, 1.0, 2.0, 3.0, 4.0, np.inf]
data['binned_bathrooms_count'] = pd.cut(data['bathrooms_count'], bins=bins_bathrooms_count, labels=labels_bathrooms_count, include_lowest= True)

# Replacing na with median:
data['review_scores_rating'] = data['review_scores_rating'].fillna(data['review_scores_rating'].median())
data['review_scores_accuracy'] = data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median())
data['review_scores_cleanliness'] = data['review_scores_cleanliness'].fillna(data['review_scores_cleanliness'].median())
data['review_scores_checkin'] = data['review_scores_checkin'].fillna(data['review_scores_checkin'].median())
data['review_scores_communication'] = data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median())
data['review_scores_location'] = data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median())
data['review_scores_value'] = data['review_scores_value'].fillna(data['review_scores_value'].median())

# Binning 'availability_365' columns
cut_labels_avail = ['Less than 2 months', '2 - 4 months', '4 - 6 months', '6 - 8 months','8 - 10 months','More than 10 months']
cut_bins_avail = [0, 60, 120, 180, 240, 300, 366]
data['binned_avail365'] = pd.cut(data['availability_365'], bins=cut_bins_avail, labels=cut_labels_avail, include_lowest= True)

# Binning 'number_of_reviews' columns
cut_labels_review = ['None', '1 - 5 reviews', 'More than 5 reviews']
cut_bins_review = [0, 1, 5, 746]
data['binned_no_of_reviews'] = pd.cut(data['number_of_reviews'], bins=cut_bins_review, labels=cut_labels_review, include_lowest= True)

# Binning 'review_scores_rating' columns
cut_labels_review_rat = ['<=90', '>90']
cut_bins_review_rat = [0, 90, 100]
data['binned_review_rating'] = pd.cut(data['review_scores_rating'], bins=cut_bins_review_rat, labels=cut_labels_review_rat, include_lowest= True)

# Binning 'review_scores_accuracy' columns
cut_labels_review_acc = ['<=9', '>9']
cut_bins_review_acc = [0, 9, 10]
data['binned_review_acc'] = pd.cut(data['review_scores_accuracy'], bins=cut_bins_review_acc, labels=cut_labels_review_acc, include_lowest= True)

# Binning 'review_scores_cleanliness' columns
cut_labels_review_clean = ['<=9', '>9']
cut_bins_review_clean = [0, 9, 10]
data['binned_review_clean'] = pd.cut(data['review_scores_cleanliness'], bins=cut_bins_review_clean, labels=cut_labels_review_clean, include_lowest= True)

# Binning 'review_scores_checkin' columns
cut_labels_review_check = ['<=9', '>9']
cut_bins_review_check = [0, 9, 10]
data['binned_review_checkin'] = pd.cut(data['review_scores_checkin'], bins=cut_bins_review_check, labels=cut_labels_review_check, include_lowest= True)

# Binning 'review_scores_communication' columns
cut_labels_review_comm = ['<=9', '>9']
cut_bins_review_comm = [0, 9, 10]
data['binned_review_comm'] = pd.cut(data['review_scores_communication'], bins=cut_bins_review_comm, labels=cut_labels_review_comm, include_lowest= True)

# Binning 'review_scores_location' columns
cut_labels_review_loc = ['<=9', '>9']
cut_bins_review_loc = [0, 9, 10]
data['binned_review_loc'] = pd.cut(data['review_scores_location'], bins=cut_bins_review_loc, labels=cut_labels_review_loc, include_lowest= True)

# Binning 'review_scores_value' columns
cut_labels_review_val = ['<=9', '>9']
cut_bins_review_val = [0, 9, 10]
data['binned_review_ val'] = pd.cut(data['review_scores_value'], bins=cut_bins_review_val, labels=cut_labels_review_val, include_lowest= True)

# Binning 'calculated_host_listings_count' columns
cut_labels_host_list_count = ['Single', 'Multiple']
cut_bins_host_list_count = [0, 1, 239]
data['binned_host_list_count'] = pd.cut(data['calculated_host_listings_count'], bins=cut_bins_host_list_count, labels=cut_labels_host_list_count, include_lowest= True)

data.to_csv('./Airbnb_V1_preprocessedv1.csv')

data['binned_host_verifications_count'].head()

