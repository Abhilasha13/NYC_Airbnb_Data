#!/usr/bin/env python
# coding: utf-8

# # NYC Airbnb Data Preprocessing and Regression

# In[1]:


# Importing the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns


# In[2]:


from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score


# In[3]:


#Importing Airbnb dataset from excel
data = pd.read_excel("/Users/ashutoshshanker/Downloads/Airbnb_V1_clean.xlsx")


# In[4]:


data.head(5)


# In[5]:


data.id.head(5)


# In[6]:


# Dataset description
data.describe().T


# In[7]:


# Columns with number and percentage of missing data
percentage_missing_data = pd.DataFrame([data.isnull().sum(), data.isnull().sum() * 100.0/data.shape[0]]).T
percentage_missing_data.columns = ['No. of Missing Data', 'Percentage of Missing data']
percentage_missing_data


# In[8]:


# Processing 'host_since' columns
data['host_since']= pd.to_datetime(data['host_since'])
data['host_since_year']=pd.DatetimeIndex(data['host_since']).year


# In[9]:


# Binning 'host_since' columns
cut_labels_hyear = ['2008-2010', '2010-2012', '2012-2014', '2014-2016','2016-2018','2018-2020']
cut_bins_hyear = [2008.0, 2010.0, 2012.0, 2014.0, 2016.0, 2018.0, 2020.0]
data['binned_hyear'] = pd.cut(data['host_since_year'], bins=cut_bins_hyear, labels=cut_labels_hyear, include_lowest= True)


# In[10]:


# Deleting rows with null values
data = data.dropna(axis=0, subset=['host_since'])


# In[11]:


# Counting number of elements in 'host_verifications' columns
host_verifications_count = []
for i in data["host_verifications"].items():
    host_verifications_count.append(i[1].count(',')+1)

data["host_verifications_count"] = host_verifications_count


# In[12]:


# Binning 'host_verifications' columns
labels_host_verifications_count = ['0-4','5-8','9 above']
bins_host_verifications_count = [0,4,8,np.inf]
data['binned_host_verifications_count'] = pd.cut(data['host_verifications_count'], bins_host_verifications_count, labels=labels_host_verifications_count, include_lowest= True)


# In[13]:


data['binned_host_verifications_count'].value_counts()


# In[14]:


# Binning 'accommodates' columns
bins_accomodate = [0, 1, 2, 3, 4, 5, 6, np.inf]
labels_accomodate =[1,2,3,4,5,6,'7 above']
data['binned_accomodate'] = pd.cut(data['accommodates'], bins_accomodate, labels=labels_accomodate)


# In[15]:


data['binned_accomodate'].value_counts()


# In[16]:


# Replacing null values with 0
data['bedrooms'] = data['bedrooms'].replace(np.nan, 0)
data['beds'] = data['beds'].replace(np.nan, 0)


# In[17]:


# Binning 'bedrooms' columns
bins_bedrooms = [0, 1, 2, 3, 4, 5,np.inf]
labels_bedrooms =[1,2,3,4,5,'5 above']
data['binned_bedrooms'] = pd.cut(data['bedrooms'], bins_bedrooms, labels=labels_bedrooms, include_lowest= True)


# In[18]:


# Binning 'beds' columns
bins_beds = [0,1, 2, 3, 4, np.inf]
labels_beds =[1,2,3,4,'5 above']
data['binned_beds'] = pd.cut(data['beds'], bins_beds, labels=labels_beds, include_lowest= True)


# In[19]:


# Binning 'price' columns
labels_price = ['<=100', '>100']
bins_price = [0, 100.00,np.inf]
data['binned_price'] = pd.cut(data['price'], bins=bins_price, labels=labels_price, include_lowest= True)


# In[20]:


data[['price','binned_price']].head()


# In[21]:


# Counting number of elements in 'amenities' columns
amenities_count = []
for i in data["amenities"].items():
    amenities_count.append(i[1].count(',')+1)

data["amenities_count"] = amenities_count
data["amenities_count"]


# In[22]:


# Binning 'amenities' columns
labels_amenities_count = ['0-10','11-20','21-30','31-40','41-50','51-60','60 above']
bins_amenities_count = [0,10,20,30,40,50,60,np.inf]
data['binned_amenities_count'] = pd.cut(data['amenities_count'], bins_amenities_count, labels=labels_amenities_count, include_lowest= True)


# In[23]:


# Splitting column 'bathrooms_text' into 'bathrooms_count' and 'bathrooms_type'
data[['bathrooms_count','bathrooms_type']] = pd.DataFrame([x.split(' ') for x in data['bathrooms_text'].tolist()])


# In[24]:


data["bathrooms_count"].unique()


# In[25]:


data["bathrooms_type"].unique()


# In[26]:


# Replacing nan and spaces with 0:
data['bathrooms_count'] = data['bathrooms_count'].replace(np.nan, 0)
data['bathrooms_count'] = data['bathrooms_count'].replace('', 0)
data['bathrooms_type'] = data['bathrooms_type'].replace('', 0)


# In[27]:


data["bathrooms_count"] = data.bathrooms_count.astype(float)


# In[28]:


# Binning 'bathrooms_type' columns
value_dict_bathrooms_type  = {'bath':'Private','private':'Private','Private':'Private','baths':'Shared','shared':'Shared','Shared':'Shared',0:'Shared'}
data["binned_bathrooms_type"] = data["bathrooms_type"].replace(value_dict_bathrooms_type)


# In[29]:


# Binning 'bathrooms_count' columns
labels_bathrooms_count = ['0-1', '1.5-2', '2.5-3', '3.5-4','4 above']
bins_bathrooms_count = [0, 1.0, 2.0, 3.0, 4.0, np.inf]
data['binned_bathrooms_count'] = pd.cut(data['bathrooms_count'], bins=bins_bathrooms_count, labels=labels_bathrooms_count, include_lowest= True)


# In[30]:


# Replacing na with median:
data['review_scores_rating'] = data['review_scores_rating'].fillna(data['review_scores_rating'].median())
data['review_scores_accuracy'] = data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median())
data['review_scores_cleanliness'] = data['review_scores_cleanliness'].fillna(data['review_scores_cleanliness'].median())
data['review_scores_checkin'] = data['review_scores_checkin'].fillna(data['review_scores_checkin'].median())
data['review_scores_communication'] = data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median())
data['review_scores_location'] = data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median())
data['review_scores_value'] = data['review_scores_value'].fillna(data['review_scores_value'].median())


# In[31]:


# Binning 'availability_365' columns
cut_labels_avail = ['Less than 2 months', '2 - 4 months', '4 - 6 months', '6 - 8 months','8 - 10 months','More than 10 months']
cut_bins_avail = [0, 60, 120, 180, 240, 300, 366]
data['binned_avail365'] = pd.cut(data['availability_365'], bins=cut_bins_avail, labels=cut_labels_avail, include_lowest= True)


# In[32]:


# Binning 'number_of_reviews' columns
cut_labels_review = ['None', '1 - 5 reviews', 'More than 5 reviews']
cut_bins_review = [0, 1, 5, 746]
data['binned_no_of_reviews'] = pd.cut(data['number_of_reviews'], bins=cut_bins_review, labels=cut_labels_review, include_lowest= True)


# In[33]:


# Binning 'review_scores_rating' columns
cut_labels_review_rat = ['<=90', '>90']
cut_bins_review_rat = [0, 90, 100]
data['binned_review_rating'] = pd.cut(data['review_scores_rating'], bins=cut_bins_review_rat, labels=cut_labels_review_rat, include_lowest= True)


# In[34]:


# Binning 'review_scores_accuracy' columns
cut_labels_review_acc = ['<=9', '>9']
cut_bins_review_acc = [0, 9, 10]
data['binned_review_acc'] = pd.cut(data['review_scores_accuracy'], bins=cut_bins_review_acc, labels=cut_labels_review_acc, include_lowest= True)


# In[35]:


# Binning 'review_scores_cleanliness' columns
cut_labels_review_clean = ['<=9', '>9']
cut_bins_review_clean = [0, 9, 10]
data['binned_review_clean'] = pd.cut(data['review_scores_cleanliness'], bins=cut_bins_review_clean, labels=cut_labels_review_clean, include_lowest= True)


# In[36]:


# Binning 'review_scores_checkin' columns
cut_labels_review_check = ['<=9', '>9']
cut_bins_review_check = [0, 9, 10]
data['binned_review_checkin'] = pd.cut(data['review_scores_checkin'], bins=cut_bins_review_check, labels=cut_labels_review_check, include_lowest= True)


# In[37]:


# Binning 'review_scores_communication' columns
cut_labels_review_comm = ['<=9', '>9']
cut_bins_review_comm = [0, 9, 10]
data['binned_review_comm'] = pd.cut(data['review_scores_communication'], bins=cut_bins_review_comm, labels=cut_labels_review_comm, include_lowest= True)


# In[38]:


# Binning 'review_scores_location' columns
cut_labels_review_loc = ['<=9', '>9']
cut_bins_review_loc = [0, 9, 10]
data['binned_review_loc'] = pd.cut(data['review_scores_location'], bins=cut_bins_review_loc, labels=cut_labels_review_loc, include_lowest= True)


# In[39]:


# Binning 'review_scores_value' columns
cut_labels_review_val = ['<=9', '>9']
cut_bins_review_val = [0, 9, 10]
data['binned_review_ val'] = pd.cut(data['review_scores_value'], bins=cut_bins_review_val, labels=cut_labels_review_val, include_lowest= True)


# In[40]:


# Binning 'calculated_host_listings_count' columns
cut_labels_host_list_count = ['Single', 'Multiple']
cut_bins_host_list_count = [0, 1, 239]
data['binned_host_list_count'] = pd.cut(data['calculated_host_listings_count'], bins=cut_bins_host_list_count, labels=cut_labels_host_list_count, include_lowest= True)


# In[41]:


t_amenities = []
for amenities in data.amenities:
    t_amenities.append(amenities)

def amenities_split(amenities):
    split_n = str(amenities).split(',')
    return split_n

t_amenities_count = []
for i in t_amenities:
    for word in amenities_split(i):
        word = re.sub(r'[^\w\s]','',word)
        word = word.lower().strip()
        t_amenities_count.append(word)


# In[42]:


from collections import Counter

Top_words=Counter(t_amenities_count).most_common()
Top_words=Top_words[0:]


# In[43]:


Top_words


# In[44]:


sub_words = pd.DataFrame(Top_words)
sub_words.rename(columns = {0:'Amenities',1:'Count'}, inplace=True)


# In[45]:


sub_words = sub_words.iloc[::-1]


# In[46]:


sub_words["Amenities"] = sub_words["Amenities"].str.replace('[^\w\s]','')


# In[47]:


#Ranking the amenities based on the number of occurance
sub_words.loc[sub_words['Amenities'] == '', 'Count'] = 0
sub_words['dense_rank'] = sub_words['Count'].rank(method='dense')


# In[48]:


sub_words


# In[49]:


# Giving weighted score to the Amenities
sub_words['score'] = sub_words['Count'] * sub_words['dense_rank']
sub_words


# In[50]:


dict_score = dict(zip(sub_words.Amenities, sub_words.score))
dict_score


# In[51]:


dict_rank = dict(zip(sub_words.Amenities, sub_words.dense_rank))
dict_rank


# In[52]:


data["amenities"].head(5)


# In[53]:


# Counting number of elements in 'amenities' columns
import re
amenities_sum = 0
dense_sum = 0
amenities_wgt_score = []

def amenities_split(amenities):
   split_n = str(amenities).split(',')
   return split_n

for i in data["amenities"]:
    
    for word in amenities_split(i):
        word = re.sub(r'[^\w\s]','',word)
        word = word.lower().strip()
        amenities_sum = amenities_sum + dict_score[word]
        dense_sum = dense_sum + dict_rank[word]
    
    wgt_score = (amenities_sum//dense_sum)
    amenities_wgt_score.append(wgt_score)

    amenities_sum = 0
    dense_sum = 0

data["amenities_wgt_score"] = amenities_wgt_score
data["amenities_wgt_score"]


# In[54]:


data.head(5)


# In[55]:


data.amenities_wgt_score.mean()


# In[56]:


sub_words.dtypes


# In[57]:


# Counting number of elements in 'amenities' columns
amenities_count = []
for i in data["amenities"].items():
    amenities_count.append(i[1].count(',')+1)

data["amenities_count"] = amenities_count
data["amenities_count"]


# In[58]:


# Binning 'amenities_wgt_score' columns
labels_amenities_wgt_score = ['0-8000', '8000-16000', '16000-24000', '24000-32000','32000 above']
bins_amenities_wgt_score = [0, 8000.0, 16000.0, 24000.0, 32000.0, np.inf]
data['binned_amenities_wgt_score'] = pd.cut(data['amenities_wgt_score'], bins=bins_amenities_wgt_score, labels=labels_amenities_wgt_score, include_lowest= True)


# In[59]:


data['binned_amenities_wgt_score']


# In[60]:


data.head(10)


# In[61]:


#data.to_excel('./Preprocessed_Airbnb_V3.1.xlsx')


# In[62]:


#data_reg = pd.read_excel("./Airbnb_V1_clean.xlsx")


# In[63]:


data_reg = data[['room_type','accommodates','neighbourhood_group_cleansed','beds','bedrooms','calculated_host_listings_count','binned_hyear','amenities_count','availability_365','review_scores_cleanliness','minimum_nights','number_of_reviews','reviews_per_month','price']]


# In[64]:


data_reg.fillna({'reviews_per_month':0}, inplace=True)


# In[65]:


percentage_missing_data = pd.DataFrame([data_reg.isnull().sum(), data_reg.isnull().sum() * 100.0/data_reg.shape[0]]).T
percentage_missing_data.columns = ['No. of Missing Data', 'Percentage of Missing data']
percentage_missing_data


# In[66]:


data_reg.head()


# In[67]:


# Correlation between the attributes
corr = data_reg.corr(method='pearson')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
data_reg.columns


# In[68]:


# One hot encoding on the catgorical columns
dataset_onehot1 = pd.get_dummies(data_reg, columns= ['room_type','neighbourhood_group_cleansed','binned_hyear'], 
                                       prefix = ['rt','ngc','byr'])


# In[69]:


#dataset_onehot1.drop(["neighbourhood_cleansed","id","name","description","host_id","host_name","host_since",
#                     "host_location","host_response_time","host_verifications","latitude","longitude",
#                      "property_type","bathrooms_text","amenities","host_since_year","binned_host_verifications_count",
#                      "binned_accomodate","binned_bedrooms","binned_beds","binned_price","binned_amenities_count",
#                      "bathrooms_type","binned_bathrooms_count","binned_avail365","binned_no_of_reviews",
#                      "binned_review_rating","binned_review_acc","binned_review_clean","binned_review_checkin",
#                      "binned_review_comm","binned_review_loc","binned_review_ val","binned_amenities_wgt_score",
#                      "host_response_rate","host_acceptance_rate","reviews_per_month"], axis=1, inplace=True)


# In[70]:


dataset_onehot1.shape


# In[71]:


X1= dataset_onehot1.loc[:, dataset_onehot1.columns != 'price']
Y1= dataset_onehot1["price"]


# In[72]:


#dataset_onehot1.to_excel('./Airbnb_onehot.xlsx')


# In[73]:


data["binned_hyear"].unique()


# In[74]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.30)


# In[75]:


# Linear Regression
reg1 = LinearRegression().fit(x_train1, y_train1)


# In[76]:


### R squared value
reg1.score(x_train1, y_train1)


# In[77]:


reg1.coef_


# In[78]:


### Predicting 
y_pred1 = reg1.predict(x_test1)


# In[79]:


Coeff1 = pd.DataFrame(columns=["Variable","Coefficient"])
Coeff1["Variable"]=x_train1.columns
Coeff1["Coefficient"]=reg1.coef_
Coeff1.sort_values("Coefficient",ascending = False)


# In[80]:


### Calculate RMSE
print(np.sqrt(metrics.mean_squared_error(y_test1,y_pred1)))
print(metrics.mean_absolute_error(y_test1, y_pred1))


# In[81]:


import statsmodels.api as sm
from scipy import stats


# In[82]:


X2 = sm.add_constant(x_train1)
est = sm.OLS(y_train1, X2)
est2 = est.fit()
print(est2.summary())


# In[83]:


regL1 = Lasso(alpha=0.01)
regL1.fit(x_train1, y_train1)


# In[84]:


regL1.score(x_train1, y_train1)


# In[85]:


y_predL1= regL1.predict(x_test1)
print(np.sqrt(metrics.mean_squared_error(y_test1,y_predL1)))
print(metrics.mean_absolute_error(y_test1, y_predL1))


# In[86]:


regL1.coef_


# In[87]:


CoeffLS1 = pd.DataFrame(columns=["Variable","Coefficients"])
CoeffLS1["Variable"]=x_train1.columns
CoeffLS1["Coefficients"]=regL1.coef_
CoeffLS1.sort_values("Coefficients", ascending = False)


# In[88]:


import time


# In[89]:


start_time = time.time()
regrRM = RandomForestRegressor(n_estimators=300)
regrRM.fit(x_train1, y_train1)
end_time = time.time()
tot_time = end_time - start_time
tot_time


# In[90]:


print(regrRM.score(x_train1, y_train1))
y_predL1= regrRM.predict(x_test1)
print(np.sqrt(metrics.mean_squared_error(y_test1,y_predL1)))
print(metrics.mean_absolute_error(y_test1, y_predL1))


# In[91]:


regrRM.feature_importances_


# In[92]:


CoeffRM1 = pd.DataFrame(columns=["Variable","FeatureImportance"])
CoeffRM1["Variable"]=x_train1.columns
CoeffRM1["FeatureImportance"]=regrRM.feature_importances_
CoeffRM1.sort_values("FeatureImportance", ascending = False)


# In[93]:


regrRM2 = RandomForestRegressor(n_estimators=300, max_depth = 50, min_samples_split = 5,min_samples_leaf =4)
regrRM2.fit(x_train1, y_train1)


# In[94]:


print(regrRM2.score(x_train1, y_train1))
y_predL1= regrRM2.predict(x_test1)
print(np.sqrt(metrics.mean_squared_error(y_test1,y_predL1)))
print(metrics.mean_absolute_error(y_test1, y_predL1))


# In[95]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
# Create the random grid
rm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[96]:


print(rm_grid)


# In[97]:


t1 = time.time()
rf2 = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
rf2_random = RandomizedSearchCV(estimator = rf2, param_distributions = rm_grid, n_iter = 180, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf2_random.fit(x_train1, y_train1)
t2 =time.time()


# In[98]:


fig, ax = plt.subplots(figsize=(12, 7))
# Remove top and right border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# Remove y-axis tick marks
ax.yaxis.set_ticks_position('none')
# Add major gridlines in the y-axis
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
ax.boxplot(dataset_onehot1["price"])


# In[99]:


plt.boxplot(dataset_onehot1["price"], notch=None, vert=None, patch_artist=None, widths=None)


# In[100]:


data['price'].corr(data['amenities_wgt_score'])


# In[101]:


X2 = sm.add_constant(x_train1)
est = sm.OLS(y_train1, X2)
est2 = est.fit()
print(est2.summary())


# In[102]:


data["binned_price"].value_counts()


# In[103]:


data[['amenities','amenities_wgt_score']].head(20)


# In[ ]:





# In[ ]:




