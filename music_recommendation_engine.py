# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:25:01 2019

@author: Hemang Thakur
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse as sp
from sklearn.metrics import pairwise as pw
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank

sns.set()
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

"""Recommendation Class for popularity based model"""

#Class for Popularity based Recommender System model
class popularity_recommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations

"""Define some useful functions for personalized hybrid recommendation model"""

#Function to create a user dictionary based on their index and number in interaction dataset
def create_user_dict(interactions):
  user_id = list(interactions.index)
  user_dict = {}
  counter = 0 
  
  for i in user_id:
    user_dict[i] = counter
    counter += 1
    
  new_dict = dict([(value, key) for key, value in user_dict.items()])
    
  return new_dict
  
#Function to create an item dictionary based on their item_id and item name  
def create_item_dict(df, id_col, name_col):
  item_dict ={}
    
  for i in range(df.shape[0]):
    item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
        
  return item_dict

#Function to produce user recommendations
def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict, threshold = 0, nrec_items = 10, show = True):
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id, :] \
                                 [interactions.loc[user_id, :] > threshold].index) \
								 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0: nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    
    if show == True:
        print("Recommended songs for UserID:", user_id)
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
            
    return return_score_list

#Function to create item-item distance embedding matrix
def create_item_emdedding_distance_matrix(model, interactions):
    
    df_item_norm_sparse = sp.csr_matrix(model.item_embeddings)
    similarities = pw.cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    
    return item_emdedding_distance_matrix

#Function to create item-item recommendation
def item_item_recommendation(item_emdedding_distance_matrix, item_id, item_dict, n_items = 10, show = True):
    
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    
    if show == True:
        print("Song of interest: {0}".format(item_dict[item_id]))
        print("Song(s) similar to the above item are as follows:-")
        counter = 1
        
        for i in recommended_items:
            print(str(counter) + '. ' +  item_dict[i])
            counter+=1
            
    return recommended_items

triplets = 'https://static.turi.com/datasets/millionsong/10000.txt'
songsData = 'https://static.turi.com/datasets/millionsong/song_data.csv'

rawData1 = pd.read_table(triplets, header=None)
rawData1.columns = ['user_id', 'song_id', 'listen_count']
rawData2 =  pd.read_csv(songsData)

#Create a new copy of the triplets dataset & change user_ids from string format to indexed values for easier computations
rawData1_userIndexed = rawData1.copy()
rawData1_userIndexed.user_id = rawData1.index + 1

#Merge the triplets data (user indexed) with songs data
rawData = pd.merge(rawData1_userIndexed, rawData2.drop_duplicates(['song_id']), on="song_id", how="left")

#Create a subset of top fifty thousand observations to work with, as the entire dataset is TOO expensive to compute on!!!
rawData = rawData.head(50000)

print(rawData.head())
print('\n', rawData.tail())
print('\n', rawData.describe(include='all'))

data = rawData.groupby(['title']).agg({'listen_count': 'count'}).reset_index()
data['percentage'] = rawData['listen_count'].div(rawData.listen_count.sum())*100
print('\n', data.sort_values(by=['listen_count'], ascending=False).head(10))

print(data['listen_count'].hist(bins=80))

users = rawData['user_id'].unique()

"""Popularity model"""

popModel = popularity_recommender()
popModel.create(rawData, 'user_id', 'title')
#popModel.create(trainData, 'user_id', 'artist_name') for popularity based recommendations by artists
print('\n', popModel.recommend(users[342]))

"""Create Interaction Matrix"""

#Create pivot table (interaction matrix) from the original dataset
x = rawData.pivot_table(index='user_id', columns='song_id', values='listen_count')
xNan = x.fillna(0)
interaction = sp.csr_matrix(xNan.values)
print(interaction)

"""Personlized hybrid model"""

hybridModel = LightFM(loss='warp-kos', n=20, k=20, learning_schedule='adadelta')
hybridModel.fit(interaction, epochs=30, num_threads=6)

"""Evaluation of the trained model"""

print('\nPrecision at K:', precision_at_k(hybridModel, interaction, k=15).mean().round(3)*100)
print('Recall at K:', recall_at_k(hybridModel, interaction, k=500).mean().round(3)*100)
print('Area under ROC curve:', auc_score(hybridModel, interaction).mean().round(3)*100)
print('Reciprocal Rank:', reciprocal_rank(hybridModel, interaction).mean().round(3)*100)

"""Recommendaing songs personally based on the user"""

#Creating user dictionary based on their index and number in the interaction matrix using recsys library
userDict = create_user_dict(interactions=x)
#print('\n', userDict)

#Creating a song dictionary based on their songID and artist name
songDict = create_item_dict(df=rawData, id_col='song_id', name_col='title')
#print('\n', songDict)

#Recommend songs using lightfm library
print('\n', sample_recommendation_user(model = hybridModel, interactions = x, user_id = 234, user_dict = userDict, item_dict = songDict, threshold = 5, nrec_items = 10,
                                      show = True))

#Recommend songs similar to a given songID
songItemDist = create_item_emdedding_distance_matrix(model=hybridModel, interactions=x)
print('\n\n', item_item_recommendation(item_emdedding_distance_matrix = songItemDist, item_id = 'SOSRCCU12A67ADA089',
                                    item_dict = songDict, n_items = 10))