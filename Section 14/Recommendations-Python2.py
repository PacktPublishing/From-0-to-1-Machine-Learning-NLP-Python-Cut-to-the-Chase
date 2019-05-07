
# coding: utf-8

# In[1]:

import numpy as np 
import pandas as pd 

# The objective is to generate some movie recommendations for a user, given movies they have already
# watched, and the ratings they gave for those movies 

# We will do this a few different ways. We'll also use Pandas, a data analysis library, for most of the 
# data preparation and analysis. 

# First let's start by downloading the dataset we'll be using. This is the  MovieLens dataset
# which is maintained by the Department of Computer Science at the University of Minnesota

# There are several datasets available of varying sizes. Let's download the 100K dataset. 
# This has 100K data points, each row is a rating given by 1 user for 1 movie at a particular date and time
# Check out the readme that comes with the data to see all the files that are provided 

# There are 2 files that we are interested in u.data - this has the userId, the movieId,
# the rating and the date that rating was given 

# u.item has a bunch of movie related details, like the title, genre, imdb url etc. 
# We'll just use this file for the movie titles 

# Pandas is a python library for data analysis in a way that's similar to 
# dataframe manipulation in R. We can read the data from a csv, write to a csv, 
# manipulate it into different shapes , subset the data based on conditions etc 

dataFile='/Users/swethakolalapudi/Downloads/ml-100k/u.data'
data=pd.read_csv(dataFile,sep="\t",header=None,
                 names=['userId','itemId','rating','timestamp'])


# This line will read the data file, it will treat it as a tab delimited file, 
# ie the columns (or values) are separated by \t
# There is no header in the file, (this is specified to Pandas by header=None)
# the names list will be used as the column names for the data
# the first column will be checked to see if it's a serial number, if yes 
# it will be automatically used as a row index. Else a row index which starts from 
# 0 will be assigned 



# In[2]:

# data is a pandas DataFrame object. There are many complex ways of indexing this 
# DataFrame aand manipulating it, subsetting it etc.. 
# head() will print the first few rows in the DataFrame
data.head()


# In[3]:

movieInfoFile='/Users/swethakolalapudi/Downloads/ml-100k/u.item'
movieInfo=pd.read_csv(movieInfoFile,sep="|", header=None, index_col=False,
                     names=['itemId','title'], usecols=[0,1])
# Here we are reading the movie data. We just care about the itemId (movieId) and 
# the title, so we are only reading the first two columns - this is specified in 
# usecols. We are explicitly passing the column names in names. Note that index_col
# is set to false. This will explicitly make sure that none of the columns in the 
# file are used to create a row index 
movieInfo.head()


# In[4]:

# Let's add the movie info ie the title to our data table. 
data=pd.merge(data,movieInfo,left_on='itemId',right_on="itemId")
# the result will be that a column 'title' will be added to our data object. 
# This line is very much like and SQL join. We are specifying the columns from 
# each table(dataframe) to join on 
data.head()


# In[5]:

# Let's now see how we can index the data in the dataframe. 
# All the values in a column can simply be indexed by the column name 
userIds=data.userId # a Pandas series object
userIds2=data[['userId']] # a Pandas DataFrame object
# Both of these are essentially the same


# In[6]:

userIds.head()


# In[7]:

userIds2.head()


# In[8]:

type(userIds)


# In[9]:

type(userIds2)


# In[10]:

# loc is a function we'll use very heavily for indexing. You can give it column 
# and row indices , or use boolean indexing. 

data.loc[0:10,['userId']]
# Give loc a list of row indices and a list of column names 








# In[11]:

toyStoryUsers=data[data.title=="Toy Story (1995)"]
# This will give us a subset dataframe with only the users who have rated Toy Story
toyStoryUsers.head()


# In[17]:

# You can sort values in the dataframe using the sort_values function 
# This function will take in the dataframe, the columns to sort on and 
# whether to sort ascending or not 
data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])

# Let's see how many users and how  many movies there are 
numUsers=max(data.userId)
numMovies=max(data.itemId)

# WE can also see how many movies were rated by each user, and the number of users
# that rated each movie 
moviesPerUser=data.userId.value_counts()
usersPerMovie=data.title.value_counts()

usersPerMovie


# numUsers

# In[18]:

# Let's write a function to find the top N favorite movies of a user 
def favoriteMovies(activeUser,N):
    #1. subset the dataframe to have the rows corresponding to the active user
    # 2. sort by the rating in descending order
    # 3. pick the top N rows
    topMovies=pd.DataFrame.sort_values(
        data[data.userId==activeUser],['rating'],ascending=[0])[:N]
    # return the title corresponding to the movies in topMovies 
    return list(topMovies.title)

print favoriteMovies(5,3) # Print the top 3 favorite movies of user 5


# In[19]:

# Let's get down to finding some recommendations now! 

# We'll start by using a neigbour based collaborative filtering model 
# The idea is to find the K Nearest neighbours of a user and 
# use their ratings to predict ratings of the active user for movies 
# they haven't rated. 

# First we'll represent each user as a vector - each element of the vector 
# will be their rating for 1 movie. Since there are 1600 odd movies in all 
# Each user will be represented by a vector that has 1600 odd values 
# When the user doesn't have any rating for a movie - the corresponding 
# element will be blank. NaN is a value in numpy that represents numbers that don't 
# exist. This is a little tricky - any operation of any other number with NaN will 
# give us NaN. So, we'll keep this mind as we manipulate the vectors 

userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                    index=['userId'], columns=['itemId'])
# pandas pivot table is very much like an excel pivot table or an SQL group by
# This will take our table which is arranged like userid, itemid, rating 
# and give us a new table in which the row index is the userId, the column idex is
# the itemId, and the value is the rating 
userItemRatingMatrix.head()


# In[20]:

# Now each user has been represented using their ratings. 
# Let's write a function to find the similarity between 2 users. We'll 
# user a correlation to do so 

from scipy.spatial.distance import correlation 
def similarity(user1,user2):
    user1=np.array(user1)-np.nanmean(user1) # we are first normalizing user1 by 
    # the mean rating of user 1 for any movie. Note the use of np.nanmean() - this 
    # returns the mean of an array after ignoring and NaN values 
    user2=np.array(user2)-np.nanmean(user2)
    # Now to find the similarity between 2 users
    # We'll first subset each user to be represented only by the ratings for the 
    # movies the 2 users have in common 
    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
    # Gives us movies for which both users have non NaN ratings 
    if len(commonItemIds)==0:
        # If there are no movies in common 
        return 0
    else:
        user1=np.array([user1[i] for i in commonItemIds])
        user2=np.array([user2[i] for i in commonItemIds])
        return correlation(user1,user2)
    


# In[24]:

# Using this similarity function, let's find the nearest neighbours of the active user
def nearestNeighbourRatings(activeUser,K):
    # This function will find the K Nearest neighbours of the active user, then 
    # use their ratings to predict the activeUsers ratings for other movies 
    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,
                                  columns=['Similarity'])
    # Creates an empty matrix whose row index is userIds, and the value will be 
    # similarity of that user to the active User
    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        # Find the similarity between user i and the active user and add it to the 
        # similarityMatrix 
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                              ['Similarity'],ascending=[0])
    # Sort the similarity matrix in the descending order of similarity 
    nearestNeighbours=similarityMatrix[:K]
    # The above line will give us the K Nearest neighbours 
    
    # We'll now take the nearest neighbours and use their ratings 
    # to predict the active user's rating for every movie
    neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    # There's something clever we've done here
    # the similarity matrix had an index which was the userId, By sorting 
    # and picking the top K rows, the nearestNeighbours dataframe now has 
    # a dataframe whose row index is the userIds of the K Nearest neighbours 
    # Using this index we can directly find the corresponding rows in the 
    # user Item rating matrix 
    predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
    # A placeholder for the predicted item ratings. It's row index is the 
    # list of itemIds which is the same as the column index of userItemRatingMatrix
    #Let's fill this up now
    for i in userItemRatingMatrix.columns:
        # for each item 
        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
        # start with the average rating of the user
        for j in neighbourItemRatings.index:
            # for each neighbour in the neighbour list 
            if userItemRatingMatrix.loc[j,i]>0:
                # If the neighbour has rated that item
                # Add the rating of the neighbour for that item
                #    adjusted by 
                #    the average rating of the neighbour 
                #    weighted by 
                #    the similarity of the neighbour to the active user
                predictedRating += (userItemRatingMatrix.loc[j,i]
                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
        # We are out of the loop which uses the nearest neighbours, add the 
        # rating to the predicted Rating matrix
        predictItemRating.loc[i,'Rating']=predictedRating
    return predictItemRating

# Let's now use these predicted Ratings to find the top N Recommendations for the
# active user 

def topNRecommendations(activeUser,N):
    predictItemRating=nearestNeighbourRatings(activeUser,10)
    # Use the 10 nearest neighbours to find the predicted ratings
    moviesAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    # find the list of items whose ratings which are not NaN
    predictItemRating=predictItemRating.drop(moviesAlreadyWatched)
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
    # This will give us the list of itemIds which are the top recommendations 
    # Let's find the corresponding movie titles 
    topRecommendationTitles=(movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)])
    return list(topRecommendationTitles.title)
    
    


# In[25]:

# Let's take this for a spin 
activeUser=5
print favoriteMovies(activeUser,5),"\n",topNRecommendations(activeUser,3)


# In[27]:

# Let's now use matrix factorization to do the same exercise ie
# finding the recommendations for a user
# The idea here is to identify some factors (these are factors which influence
# a user'r rating). The factors are identified by decomposing the 
# user item rating matrix into a user-factor matrix and a item-factor matrix
# Each row in the user-factor matrix maps the user onto the hidden factors
# Each row in the product factor matrix maps the item onto the hidden factors
# This operation will be pretty expensive because it will effectively give us 
# the factor vectors needed to find the rating of any product by any user 
# (in the  previous case we only did the computations for 1 user)

def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):
    # R is the user item rating matrix 
    # K is the number of factors we will find 
    # We'll be using Stochastic Gradient descent to find the factor vectors 
    # steps, gamma and lamda are parameters the SGD will use - we'll get to them
    # in a bit 
    N=len(R.index)# Number of users
    M=len(R.columns) # Number of items 
    P=pd.DataFrame(np.random.rand(N,K),index=R.index)
    # This is the user factor matrix we want to find. It will have N rows 
    # on for each user and K columns, one for each factor. We are initializing 
    # this matrix with some random numbers, then we will iteratively move towards 
    # the actual value we want to find 
    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
    # This is the product factor matrix we want to find. It will have M rows, 
    # one for each product/item/movie. 
    for step in xrange(steps):
        # SGD will loop through the ratings in the user item rating matrix 
        # It will do this as many times as we specify (number of steps) or 
        # until the error we are minimizing reaches a certain threshold 
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    # For each rating that exists in the training set 
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    # This is the error for one rating 
                    # ie difference between the actual value of the rating 
                    # and the predicted value (dot product of the corresponding 
                    # user factor vector and item-factor vector)
                    # We have an error function to minimize. 
                    # The Ps and Qs should be moved in the downward direction 
                    # of the slope of the error at the current point 
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    # Gamma is the size of the step we are taking / moving the value
                    # of P by 
                    # The value in the brackets is the partial derivative of the 
                    # error function ie the slope. Lamda is the value of the 
                    # regularization parameter which penalizes the model for the 
                    # number of factors we are finding. 
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        # At the end of this we have looped through all the ratings once. 
        # Let's check the value of the error function to see if we have reached 
        # the threshold at which we want to stop, else we will repeat the process
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    #Sum of squares of the errors in the rating
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        print step
    return P,Q

# Let's call this function now 
(P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=100)
# Ideally you should run this over the entire matrix for a few 1000 steps, 
# This will be pretty expensive computationally. For now lets just do it over a 
# part of the rating matrix to see how it works. We've kept the steps to 100 too. 
  
    


# In[ ]:

# This will take a while to run, you can see which step its currently on 


# In[29]:

# Let's quickly use these ratings to find top recommendations for a user 
activeUser=1
predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
# We found the ratings of all movies by the active user and then sorted them to find the top 3 movies 
topRecommendationTitles=movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)]
print list(topRecommendationTitles.title)


# In[ ]:

# Let's now find Association rules from the Movielens dataset

# Association rules normally make sense with purchases / transactions datasets
# Here the rule won't have much meaning, except to say a person who watches
# movie a will also be likely to have watched movie b 
# We'll just quickly implement the apriori algorithm on this dataset to see how 
# it works 


# In[33]:

import itertools 
# This module will help us generate all permutations of movies
# We'll use that to find the possible rules and then filter for those with 
# the required confidence

allitems=[]

def support(itemset):
    userList=userItemRatingMatrix.index
    nUsers=len(userList)
    ratingMatrix=userItemRatingMatrix
    for item in itemset:
        ratingMatrix=ratingMatrix.loc[ratingMatrix.loc[:,item]>0]
        #Subset the ratingMatrix to the set of users who have rated this item 
        userList=ratingMatrix.index
    # After looping through all the items in the set, we are left only with the
    # users who have rated all the items in the itemset
    return float(len(userList))/float(nUsers)
# Support is the proportion of all users who have watched this set of movies 

minsupport=0.3
for item in list(userItemRatingMatrix.columns):
    itemset=[item]
    if support(itemset)>minsupport:
        allitems.append(item)
# We are now left only with the items which have been rated by atleast 30% of 
#the users
        




# In[34]:

len(allitems)


# In[35]:

# 47 of the movies were watched by atleast 30% of the users. From these movies
# we'll generate rules and test again for support and confidence
minconfidence=0.1
assocRules=[]
i=2
for rule in itertools.permutations(allitems,2):
    #Generates all possible permutations of 2 items from the remaining
    # list of 47 movies 
    from_item=[rule[0]]
    to_item=rule
    # each rule is a tuple of 2 items 
    confidence=support(to_item)/support(from_item)
    if confidence>minconfidence and support(to_item)>minsupport:
        assocRules.append(rule)


# In[ ]:

# This will generate all possible 2 item rules which satisfy the support and 
# confidence constraints. 
# You can continue on and write a similar bit of code for finding 3 item rules 
# or even n item rules. At each step make sure that every rule satisfies minconfidence
# and minsupport


# In[36]:

assocRules


# In[ ]:



