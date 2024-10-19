#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries
# 

# In[52]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numbers as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


# ### Import Dataset
# 

# In[53]:


df=pd.read_csv('dataset.csv',header=None)


# In[54]:


df.head(2)


# In[55]:


df.columns=['user_id','prod_id','rating','timestamp']


# In[56]:


df.head(2)


# In[57]:


df=df.drop('timestamp',axis=1)


# In[58]:


df.head(2)


# In[59]:


database=df.copy(deep=True)


# In[60]:


database.head(2)


# ### Exploratory Data Aanlysis
# 

# In[61]:


rows,coloumns=database.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {coloumns}")


# In[62]:


database.info()


# In[63]:


database.isna().sum()


# In[64]:


database['rating'].describe()


# In[65]:


plt.figure(figsize=(12,6))
database['rating'].value_counts(1).plot(kind='bar')
plt.show()


# In[66]:


num_users=database['user_id'].nunique()
num_products=database['prod_id'].nunique()

print(f"Number of unique users: {num_users}")

print(f"Number of unique products: {num_products}")


# #### The code identifies and stores the top 10 users who have provided the most ratings in the database, along with the count of their ratings.

# In[67]:


# Top 10 users with max ratings given to products
most_rated_users =database.groupby('user_id').size().sort_values(ascending=False)[:10]
# print(most_rated_users)
most_rated_users


# ## Pre-Processing
# 

# #### The code creates a new DataFrame, new_database, which contains only the rows from database corresponding to users who have rated 50 or more items. This is useful for analyzing a subset of users with a significant amount of interaction in the dataset.
# 

# In[68]:


counts=database['user_id'].value_counts()
# print(counts)
new_database=database[database['user_id'].isin(counts[counts>=50].index)]


# In[69]:


print('The number of observations in the final data =', len(new_database))
print('Number of unique USERS in the final data = ', new_database['user_id'].nunique())
print('Number of unique PRODUCTS in the final data = ', new_database['prod_id'].nunique())


# In[70]:


### Checking the density of the rating matrix
final_new_database =new_database.pivot(index='user_id', columns='prod_id',values='rating').fillna(0)
print("Shape of the final_new_database is: ", final_new_database.shape)


# In[71]:


import numpy as np
num_ratings=np.count_nonzero(final_new_database)
print("Num of non-zero ratings of the final_new_database: ",num_ratings)


# In[72]:


#Finding the possible number of ratings as per the number of users and products
possible_num_of_ratings = final_new_database.shape[0] * final_new_database.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)


# In[73]:


#Density of ratings
density = (num_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))


# In[74]:


final_new_database.head()


# ## Rank Based Recommendation System
# 

# In[75]:


print(new_database['rating'].dtype)


# In[76]:


print(new_database['rating'].isnull().sum())


# In[77]:


#Calculate the average rating for each product 
average_rating = new_database.groupby('prod_id')['rating'].mean()

#Calculate the count of ratings for each product
count_rating = new_database.groupby('prod_id')['rating'].count()

#Create a dataframe with calculated average and count of ratings
final_rating = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})

#Sort the dataframe by average of ratings
final_rating = final_rating.sort_values(by='avg_rating',ascending=False)

final_rating.head()


# In[78]:


#defining a function to get the top n products based on highest average rating and minimum interactions
def top_n_products(final_rating, n, min_interaction):
    
    #Finding products with minimum number of interactions
    recommendations = final_rating[final_rating['rating_count']>min_interaction]
    
    #Sorting values w.r.t average rating 
    recommendations = recommendations.sort_values('avg_rating',ascending=False)
    
    return recommendations.index[:n]


# ### Recommending top 5 products with 50 minimum interactions based on popularity
# 

# In[79]:


list(top_n_products(final_rating, 5, 50))


# ### Recommending top 5 products with 100 minimum interactions based on popularity
# 

# In[80]:


list(top_n_products(final_rating, 5, 100))


# # Collaborative Filtering based Recommendation System
# 

# In[81]:


final_new_database.head()


# In[82]:


final_new_database['user_index']=np.arange(0, final_new_database.shape[0])
final_new_database.set_index(['user_index'],inplace=True)

final_new_database.head()


# ### Function to find the Similar Users and their similarity scores
# 

# In[83]:


# defining a function to get similar users
def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(0, interactions_matrix.shape[0]): #  .shape[0] gives number of rows
        
        #finding cosine similarity between the user_id and each user
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        
        #Appending the user and the corresponding similarity score with user_id as a tuple
        similarity.append((user,sim))
        
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity] #Extract the user from each tuple in the sorted list
    similarity_score = [tup[1] for tup in similarity] ##Extracting the similarity score from each tuple in the sorted list
   
    #Remove the original user and its similarity score and keep only other similar users 
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])
       
    return most_similar_users, similarity_score


# In[84]:


similar=similar_users(3,final_new_database)[0][0:10]


# In[85]:


similar


# In[86]:


similar_users(3,final_new_database)[1][0:10]


# ### Function to recommend products
# 

# In[87]:


# defining the recommendations function to get recommendations by using the similar users' preferences
def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))
    recommendations = []
    
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            
            #Finding 'n' products which have been rated by similar users but not by the user_id
            similar_user_prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    
    return recommendations[:num_of_products]


# ### Recommend 5 products to user index 3 based on similarity based collaborative filtering
# 

# In[88]:


recommendations(3,5,final_new_database)


# ### Recommend 5 products to user index 1521 based on similarity based collaborative filtering
# 

# In[89]:


recommendations(1521,5,final_new_database)

