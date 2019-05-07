
# coding: utf-8

# In[3]:

#####################################################################
# Step 1: Accept a search term from the user and download 
# the last 100 tweets from that search term 
#####################################################################

# Install the python-twitter module. Unfortunately this module works only with Python 2 currently
# and the Python 3 support is still under development. There are other modules that are similar though and
# some are listed on the Twitter API documentation website 
# https://dev.twitter.com/overview/api/twitter-libraries
 

# Otherwise, you can just go ahead and use !pip install python-twitter to install python-twitter for Python 2.  
# This is a module that provides a python like interface to the Twitter API. The Twitter API is 
# fairly straightforward to use if you have used REST APIs before. A REST API provides information 
# in the form of a JSON which your application will have to parse once you get it. python-twitter 
# does this work for you and abstracts you from having to know the nitty-gritty of the Twitter API. In case the 
# module that you are using provides you a json output; you can use the json library to parse the tweets. This would
# be an additional step that we have not shown you in our script. 


import twitter

# Here we are importing the python-twitter module. (The library you import is called twitter, this 
# is a bit peculiar, but just remember that you will install python-twitter but in the import statement
# import twitter)

# The module provides an API object which has methods to get information from the Twitter API. To see 
# the complete documentation type pydoc twitter.Api at the command prompt in your terminal. This will 
# show you all the methods available, including those for fetching a user's statuses, a user's followers,
# statuses for a particular search term etc 
# You can even post a status message to Twitter using this Api object but let's not go there right now :) 

# The Api object will need your Twitter API key/access credentials. Get these by registering your app 
# at https://apps:twitter.com/app/new 

api = twitter.Api(consumer_key='tpAestpXtM2pYAlomfZr6LN7d',
                 consumer_secret='MCQ1aVPypaBOZIlg7MDp36znULAIcmf9Cj8xfxodyVyLpILpQu',
                 access_token_key='124163864-koQiHbqAF1QvLUzGqMb2ITvWk60jaa5yOsgJeaT7',
                 access_token_secret='qdMSjnab0O49k1pnck0fgtVQre60VN7pb0qkSC2vSYwJE')

# To see if this worked, use the command below, it will print out a bunch of details about your user account
# and that's how you know you're all set to use the API
print(api.VerifyCredentials())



# In[4]:

# We're all set with our API 
# Now we'll set up a function to accept a search term and then fetch the tweets for that term 

def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=100)
        # This will return a list with twitter.Status objects. These have attributes for 
        # text, hashtags etc of the tweet that you are fetching. 
        # The full documentation again, you can see by typing pydoc twitter.Status at the 
        # command prompt of your terminal 
        print "Great! We fetched "+str(len(tweets_fetched))+" tweets with the term "+search_string+"!!"
        # We will fetch only the text for each of the tweets, and since these don't have labels yet, 
        # we will keep the label empty 
        return [{"text":status.text,"label":None} for status in tweets_fetched]
    except:
        print "Sorry there was an error!"
        return None
    
search_string=input("Hi there! What are we searching for today?")
testData=createTestData(search_string)

# Let's try that out
    

    


# In[5]:

testData[0:9]


# In[ ]:

###############################################################
# Step 2: Classify each of the 100 tweets as positive or negative
################################################################

# 2a. Download a corpus of tweets to use as training data
# We'll use Niek Sanders's Tweet Sentiment Corpus. He has kindly provided 5000+ labelled tweets
# WE can download a csv from his website with the tweets. But there is a catch, Twitter only allows 
# sharing of the tweet_id, so we'll have to fetch the text for each tweet id from the twitter API 

# We'll write a function that will read the csv we got from his website, for each tweet id in the 
# csv we'll download the tweet text and then write it back to another csv 

def createTrainingCorpus(corpusFile,tweetDataFile):
    import csv
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})
    # We now have a list with a dictionary for each row in Sanders's csv
    # Next let's iterate through that list and fetch the text for each tweet_id
    
    # If you try to download more than 180 tweets/15 mins, Twitter will rate limit you. So, use a delay
    # to avoid being rate limited. But this means it will take 10+ hours to download all 5000 tweets 
    # We'll show you the code to download all 5000 tweets, but for now, we'll work with a smaller corpus
    # so we won't have to wait 10 hours to see our code run :) 
    
    # To download the full corpus 
    import time 
    rate_limit=180
    sleep_time=900/180
    
    trainingData=[]
    for tweet in corpus:
        try:
            status=api.GetStatus(tweet["tweet_id"])
            #Returns a twitter.Status object 
            print "Tweet fetched" + status.text
            tweet["text"]=status.text
            #tweet is a dictionary which already has tweet_id and label (positive/negative/neutral)
            # Add another attribute now, the tweet text 
            trainingData.append(tweet)
            time.sleep(sleep_time) # to avoid being rate limited 
        except: 
            continue
    # Once the tweets are downloaded write them to a csv, so you won't have to wait 40 hours 
    # every time you run this code :) 
    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except Exception,e:
                print e
    return trainingData


# In[17]:


# Let's now write a separate function to download just 50 tweets for each label 

def createLimitedTrainingCorpus(corpusFile,tweetDataFile):
    import csv
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})
    # We now have a list with a dictionary for each row in Sanders's csv
    # Next let's iterate through that list and fetch the text for each tweet_id
    
    # If you try to download more than 180 tweets/15 min, Twitter will rate limit you. So, use a delay
    # to avoid being rate limited. But this means it will take 10+ hours to download all 5000 tweets 
    # We'll show you the code to download all 5000 tweets, but for now, we'll work with a smaller corpus
    # so we won't have to wait 10 hours to see our code run :) 
    
    # To download the full corpus 
    
    trainingData=[]
    for label in ["positive","negative"]:
        i=1
        for tweet in corpus:
            if tweet["label"]==label and i<=50:
                try:
                    status=api.GetStatus(tweet["tweet_id"])
                    #Returns a twitter.Status object 
                    print "Tweet fetched" + status.text
                    tweet["text"]=status.text
                    #tweet is a dictionary which already has tweet_id and label (positive/negative/neutral)
                    # Add another attribute now, the tweet text 
                    trainingData.append(tweet)
                    i=i+1
                except Exception, e: 
                    print e
                    
    # Once the tweets are downloaded write them to a csv, so you won't have to wait 10 hours 
    # every time you run this code :) 
    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        # We'll add a try catch block here so that we still get the training data even if the write 
        # fails 
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except Exception,e:
                print e
    return trainingData

corpusFile="/Users/swethakolalapudi/Downloads/sanders-twitter-0.2/corpus.csv"
tweetDataFile="/Users/swethakolalapudi/Downloads/sanders-twitter-0.2/tweetDataFile.csv"

trainingData=createLimitedTrainingCorpus(corpusFile,tweetDataFile)
# This will have saved our 150 tweets to a file and also returned a list with all the tweet data we 
# need for training


# In[20]:

# 2b. A class to preprocess all the tweets, both test and training
# We will use regular expressions and NLTK for preprocessing 
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 


class PreProcessTweets:
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])
        
    def processTweets(self,list_of_tweets):
        # The list of tweets is a list of dictionaries which should have the keys, "text" and "label"
        processedTweets=[]
        # This list will be a list of tuples. Each tuple is a tweet which is a list of words and its label
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self,tweet):
        # 1. Convert to lower case
        tweet=tweet.lower()
        # 2. Replace links with the word URL 
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)     
        # 3. Replace @username with "AT_USER"
        tweet=re.sub('@[^\s]+','AT_USER',tweet)
        # 4. Replace #word with word 
        tweet=re.sub(r'#([^\s]+)',r'\1',tweet)
        # You can do further cleanup as well if you like, replace 
        # repetitions of characters, for ex: change "huuuuungry" to "hungry"
        # We'll leave that as an exercise for you on regular expressions
        tweet=word_tokenize(tweet)
        # This tokenizes the tweet into a list of words 
        # Let's now return this list minus any stopwords 
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessor=PreProcessTweets()
ppTrainingData=tweetProcessor.processTweets(trainingData)
ppTestData=tweetProcessor.processTweets(testData)


# In[41]:

# 2c. Extract features and train your classifier

# We'll use two methods - Naive Bayes and Support Vector Machines 

import nltk 
# Naive Bayes Classifier - We'll use NLTK's built in classifier to perform the classification

# First build a vocabulary 
def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment) in ppTrainingData:
        all_words.extend(words)
    # This will give us a list in which all the words in all the tweets are present
    # These have to be de-duped. Each word occurs in this list as many times as it 
    # appears in the corpus 
    wordlist=nltk.FreqDist(all_words)
    # This will create a dictionary with each word and its frequency
    word_features=wordlist.keys()
    # This will return the unique list of words in the corpus 
    return word_features

# NLTK has an apply_features function that takes in a user-defined function to extract features 
# from training data. We want to define our extract features function to take each tweet in 
# The training data and represent it with the presence or absence of a word in the vocabulary 

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
        # This will give us a dictionary , with keys like 'contains word1' and 'contains word2'
        # and values as True or False 
    return features 

# Now we can extract the features and train the classifier 
word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)
# apply_features will take the extract_features function we defined above, and apply it it 
# each element of ppTrainingData. It automatically identifies that each of those elements 
# is actually a tuple , so it takes the first element of the tuple to be the text and 
# second element to be the label, and applies the function only on the text 

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
# We now have a classifier that has been trained using Naive Bayes

# Support Vector Machines 
from nltk.corpus import sentiwordnet as swn
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 

# We have to change the form of the data slightly. SKLearn has a CountVectorizer object. 
# It will take in documents and directly return a term-document matrix with the frequencies of 
# a word in the document. It builds the vocabulary by itself. We will give the trainingData 
# and the labels separately to the SVM classifier and not as tuples. 
# Another thing to take care of, if you built the Naive Bayes for more than 2 classes, 
# SVM can only classify into 2 classes - it is a binary classifier. 

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingData]
# Creates sentences out of the lists of words 

vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
# We now have a term document matrix 
vocabulary=vectorizer.get_feature_names()

# Now for the twist we are adding to SVM. We'll use sentiwordnet to add some weights to these 
# features 

swn_weights=[]

for word in vocabulary:
    try:
        # Put this code in a try block as all the words may not be there in sentiwordnet (esp. Proper
        # nouns). Look for the synsets of that word in sentiwordnet 
        synset=list(swn.senti_synsets(word))
        # use the first synset only to compute the score, as this represents the most common 
        # usage of that word 
        common_meaning =synset[0]
        # If the pos_Score is greater, use that as the weight, if neg_score is greater, use -neg_score
        # as the weight 
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight=common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight=-common_meaning.neg_score()
        else: 
            weight=0
    except: 
        weight=0
    swn_weights.append(weight)


# Let's now multiply each array in our original matrix with these weights 
# Initialize a list

swn_X=[]
for row in X: 
    swn_X.append(np.multiply(row,np.array(swn_weights)))
# Convert the list to a numpy array 
swn_X=np.vstack(swn_X)


# We have our documents ready. Let's get the labels ready too. 
# Lets map positive to 1 and negative to 2 so that everything is nicely represented as arrays 
labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingData]
y=np.array(labels)

# Let's now build our SVM classifier 
from sklearn.svm import SVC 
SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)


# In[42]:

# Step 2d: Run the classifier on the 100 downloaded tweets 

# First Naive Bayes 
NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]

# Now SVM 
SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])
    # predict() returns  a list of numpy arrays, get the first element of the first array 
    # there is only 1 element and array






# In[43]:

# Step 3 : GEt the majority vote and print the sentiment 

if NBResultLabels.count('positive')>NBResultLabels.count('negative'):
    print "NB Result Positive Sentiment" + str(100*NBResultLabels.count('positive')/len(NBResultLabels))+"%"
else: 
    print "NB Result Negative Sentiment" + str(100*NBResultLabels.count('negative')/len(NBResultLabels))+"%"
    
    
    
    
if SVMResultLabels.count(1)>SVMResultLabels.count(2):
    print "SVM Result Positive Sentiment" + str(100*SVMResultLabels.count(1)/len(SVMResultLabels))+"%"
else: 
    print "SVM Result Negative Sentiment" + str(100*SVMResultLabels.count(2)/len(SVMResultLabels))+"%"
  
  


# In[44]:

testData[0:10]


# In[33]:

NBResultLabels[0:10]


# In[34]:

SVMResultLabels[0:10]


# In[ ]:

#Looks like most of these tweets are actually neutral , And our SVM is classifying them as -ve,
# But it classified the positive tweet correctly. 

# A few next steps possible here 
# Remove all neutral words according to sentiwordnet from the vocabulary. 
# Look at things like ALL CAPS , emoticons etc 

# GEt a corpus with more varied tweets (This one has only tech related tweets, so it works for our 
# search term (Apple) but might not for others. )

