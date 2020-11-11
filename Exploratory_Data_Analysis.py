#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

data = pd.read_excel("./Airbnb_V1_clean.xlsx")

# Summary statistics of dataset:
# Dataset description
data.describe().T

# Count of listings in different neighborhoods
data.neighbourhood_group_cleansed.value_counts()

t_amenities = []
for amenities in data.amenities:
    t_amenities.append(amenities)
    
def amenities_split(amenities):
    split_n = str(amenities).split(',')
    return split_n

t_amenities_count = []
for i in t_amenities:
    for word in amenities_split(i):
        word = word.lower()
        t_amenities_count.append(word)

from collections import Counter
#top 25 used words by amenities to name their listing
Top_25_words=Counter(t_amenities_count).most_common()
Top_25_words=Top_25_words[0:25]

# Count of occurence of each amenity in the dataset:
Top_25_words

#putting the findings in dataframe for further visualizations
sub_words = pd.DataFrame(Top_25_words)
sub_words.rename(columns = {0:'Amenities',1:'Count'}, inplace=True)

# Plot representing the top 25 most common amenity:
viz_1=sns.barplot(x='Amenities', y='Count', data=sub_words)
viz_1.set_title('Counts of the top 25 used Amenities for listing names')
viz_1.set_ylabel('Count of words')
viz_1.set_xlabel('Amenities')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=80)

# Word Cloud representing the frequency of each amenity:
plt.subplots(figsize=(10,6))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(data.amenities))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('amenities.png')
plt.show()

# Airbnb room type vs price of any listing
plt.figure(figsize=(6,4))
sns.barplot(x='room_type', y='price', data=data)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price",size=15, weight='bold')

# Count of room_type available in neighborhood locations. 
plt.figure(figsize=(6,4))
sns.countplot(x = 'room_type',hue = "neighbourhood_group_cleansed",data = data)
plt.title("Room types occupied by the neighbourhood_group")
plt.show()

# Representation of available Airbnb listings in different neighbourhood_group 
plt.figure(figsize=(10,6))
sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group_cleansed)
plt.ioff()

# It can be observed that low cost rooms or rooms in range 0-50 $ have more reviews.
plt.figure(figsize=(10,6))
data['number_of_reviews'].plot(kind='hist')
plt.xlabel("price")
plt.ioff()
plt.show()

plt.figure(figsize=(20,15))
sns.scatterplot(x="room_type", y="price",
            hue="neighbourhood_group_cleansed", size="neighbourhood_group_cleansed",
            sizes=(50, 200), palette="Dark2", data=data)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price vs Neighbourhood Group",size=15, weight='bold')

# Area wise distribution of price shows that Manhattan has expensive and Staten Island has low priced rooms
plt.figure(figsize=(10,6))
sns.distplot(data[data.neighbourhood_group_cleansed=='Manhattan'].price,color='maroon',hist=False,label='Manhattan')
sns.distplot(data[data.neighbourhood_group_cleansed=='Brooklyn'].price,color='black',hist=False,label='Brooklyn')
sns.distplot(data[data.neighbourhood_group_cleansed=='Queens'].price,color='green',hist=False,label='Queens')
sns.distplot(data[data.neighbourhood_group_cleansed=='Staten Island'].price,color='blue',hist=False,label='Staten Island')
sns.distplot(data[data.neighbourhood_group_cleansed=='Bronx'].price,color='orange',hist=False,label='Bronx')
plt.title('Borough wise price destribution for price<2000')
plt.xlim(0,2000)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x = 'room_type',hue = "neighbourhood_group_cleansed",data = data)
plt.title("Room types occupied by the neighbourhood_group")
plt.show()

