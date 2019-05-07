
# coding: utf-8

# In[ ]:

# Given a passenger on the Titanic and their attributes - like age, sex, socio-economic statusl the objective is to 
# predict whether they would have survived or not 

# This is one of the challenges on Kaggle - a competitive online Data Science community
# We'll fetch the data from the Kaggle website and in the end we'll submit our results to see where we stand
# among the Data Science heavyweights :) 

# The csv contains the following details 
# passengerid      The row/serial number 
# survival         Survival (0 = No; 1= Yes)
# pclass           Passenger Class(1 = 1st; 2 = 2nd; 3=3rd) This can be a proxy for socio-economic status
# name             Name
# sex              Sex (male/female) This will need to be converted to a numerical vairable. More on this later
# age              Age (in years); Fractional if age<1 and represented as xx.5 if its an estimated age
# sibsp            Number of Siblings/Spouses Aboard
# parch            Number of Parents/Children Aboard
# ticket           Ticket Number 
# fare             Passenger Fare
# cabin            Cabin 
# embarked         Port of Embarkation (C=Cherbourg; Q=Queenstown; S=Southampton)

# We'll use sci-kit learn's DecisionTreeClassifier. This classifier takes only numeric variables as part of the feature 
# vector. We will have to convert the categorical variables like Sex and Port of Embarkation to numeric variables
# WE'll do this by mapping each of the labels to a number 


# In[15]:

import csv 

def transformDataTitanic(trainingFile, features):
    # This function will read the data in the training file and transform it 
    # into a list of lists. 
    # Let's initialize a variable to hold this list of lists
    transformData=[]
    # The function will also return a list with the labels (Survived (0/1)) for
    # each of the passengers
    labels = []
    # Now we'll set up a couple of maps. These will be used to convert the 
    # categorical variables like gender to numeric variables 
    
    genderMap = {"male":1,"female":2,"":""} # We include a key in the map for missing values
    embarkMap = {"C":1,"Q":2,"S":3,"":""}
    # Notice how the map contains a key for blanks. The csv we got has a lot of missing values
    # and how you deal with these could have a big impact on your ability to predict accurately. 
    # For now we are just going to ignore any passengers who have missing values in any of the features
    # We'll initialize a blank string to perform this check before processing a passenger
    blank=""
    # Now we are finally ready to read the csv file
    with open(trainingFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        # lineNum will help us keep track of which row we are in 
        for row in lineReader:
            if lineNum==1:
                # if it's the first row, just store the names in the header in a list. The features
                # That are passed in to our function will be a subset of this list
                # PassengerID, Survived, PClass, Name, Sex, Age, Sibsp, Parch, Ticket, Fare, Cabin,
                # Embarked
                header = row
            else: 
                allFeatures=map(lambda x:genderMap[x] if row.index(x)==4
                               else embarkMap[x] if row.index(x)==11 else x, row)
                # allFeatures is a list where we have converted the categorical variables to 
                # numerical variables
                featureVector = [allFeatures[header.index(feature)] for feature in features]
                # featureVector is a subset of allFeatures, it contains only those features
                # that are specified by us in the function argument
                if blank not in featureVector:
                    transformData.append(featureVector)
                    labels.append(int(row[1]))
                    # if the featureVector contains missing values, skip it, else add the featureVector
                    # to our transformedData and the corresponding label to the list of labels
            lineNum=lineNum+1
        return transformData,labels
    # return both our list of feature vectors and the list of labels 
                
            
    


# In[38]:

# Let's take this for a spin now
trainingFile="/Users/swethakolalapudi/Desktop/Kaggle/Titanic/train.csv"
features=["Pclass","Sex","Age","SibSp","Fare"]
trainingData=transformDataTitanic(trainingFile,features)


# In[52]:

# We are now ready to train our Decision Tree classifier
from sklearn import tree
import numpy as np
clf=tree.DecisionTreeClassifier(max_leaf_nodes=20)
X=np.array(trainingData[0])
y=np.array(trainingData[1])
clf=clf.fit(X,y)
# The fit method takes in 2 numpy arrays, one with the feature vectors and the other with the 
# labels. It uses a modified version of the CART algorithm and uses Gini Impurity as the 
# measure to base the split of. The idea of Gini Impurity is that if we stop the classifier
# after splitting by that attribute, What would be the probability of a false label 
# You can change the "criterion" used by the classifier to use information gain instead
# clf=tree.DecisionTreeClassifier(criterion="entropy")


# In[45]:

# Our Decision tree has now been created. To see it represented visually, we'll use a 
# software called graphviz. You can download and install the graphviz application. Using
# sci-kit learn we can export our tree to a file with .dot extension which will open up in 
# graphviz. If you open the .dot file in a plain text editor - it will look very much like 
# a markup language like xml

with open("titanic.dot","w") as f:
    f = tree.export_graphviz(clf,
                            feature_names=features,
                            class_names=["Dead","Survived"],
                            filled=True, rounded=True,special_characters=True,out_file=f)


# In[ ]:

# AS we saw , the tree created is huge, it keeps going until it cannot split the training 
# data any further. It's very hard to make sense of what is going on in this tree, and 
# it's most likely overfitted.
# The DecisionTreeClassifier() has many attributes you can use to control the tree
# max_depth : The max allowed depth of the tree
# max_leaf_nodes : The max number of leaf nodes allowed 
# min_samples_split: The minimum number of samples to perform a split, if a subset has lesser
# samples, it's made a leaf node with whichever label has the max count


# In[28]:

# The tree visually tells you which are the more important features. The classifier has an 
# attribute which tells us the relative importance of features using a metric known as 
# Gini Importance 
clf.feature_importances_


# In[46]:

# Let's see how this tree performs when you run it on the test data from Kaggle. 
# We'll use our classifier to predict whether the passenger survived. Then we'll submit
# the results and see how it goes ! 
def transformTestDataTitanic(testFile,features):
    # We'll do a similar transformation on the test data, ie pick the specified features and
    # map categorical variables to numerical variables 
    # In this case we need to keep track of the passenger ids so we can write them back to the 
    # csv that we'll submit to Kaggle
    transformData=[]
    ids=[]
    genderMap={"male":1,"female":2,"":""}
    embarkMap={"C":1,"Q":2,"S":3,"":""}
    blank=""
    with open(testFile,"rb") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        for row in lineReader:
            if lineNum==1:
                header=row
            else: 
                allFeatures=map(lambda x:genderMap[x] if row.index(x)==3 else embarkMap[x] 
                               if row.index(x)==10 else x,row)
                featureVector=[allFeatures[header.index(feature)] for feature in features]
                # note that the test csv does not contain a column with the labels (Survived (1/0))
                # We'll have to deal with blanks a little differently here
                # We can't just ignore all the passengers with missing values as the Kaggle 
                # submission needs you to predict the outcome even for passengers whose attribute data
                # is missing
                # So, for now let's use a default value whenever we encounter a blank
                featureVector=map(lambda x:0 if x=="" else x, featureVector)
                transformData.append(featureVector)
                ids.append(row[0])
            lineNum=lineNum+1 
    return transformData,ids


# In[47]:

def titanicTest(classifier,resultFile,transformDataFunction=transformTestDataTitanic):
    # This function will take our classifier and run it on the test data
    testFile="/Users/swethakolalapudi/Desktop/Kaggle/Titanic/test.csv"
    testData=transformDataFunction(testFile,features)#call the function we just wrote
    result=classifier.predict(testData[0])
    with open(resultFile,"wb") as f:
        ids=testData[1]
        lineWriter=csv.writer(f,delimiter=',',quotechar="\"")
        lineWriter.writerow(["PassengerId","Survived"])#The submission file needs to have a header
        for rowNum in range(len(ids)):
            try:
                lineWriter.writerow([ids[rowNum],result[rowNum]])
            except Exception,e:
                print e

# Let's take this for a spin! 
resultFile="/Users/swethakolalapudi/Desktop/Kaggle/Titanic/result1.csv"
titanicTest(clf,resultFile)


# In[48]:

# So our current strategy is bettern than predicting that every passenger has died. But not by much. 
# One of the reasons this performs so badly is because of the way we dealt with the missing values 
# Let's now use a slightly more intelligent strategy to replace the missing values
def transformTestDataTitanicv2(testFile,features):
    # We'll do a similar transformation on the test data, ie pick the specified features and
    # map categorical variables to numerical variables 
    # In this case we need to keep track of the passenger ids so we can write them back to the 
    # csv that we'll submit to Kaggle
    transformData=[]
    ids=[]
    # This time we'll map each missing value with the most common value or the average value of that variable
    genderMap={"male":1,"female":2,"":1} # Map blanks to males
    embarkMap={"C":1,"Q":2,"S":3,"":3} # Map the default port of embarkation to Southampton
    blank=""
    with open(testFile,"rb") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        for row in lineReader:
            if lineNum==1:
                header=row
            else: 
                allFeatures=map(lambda x:genderMap[x] if row.index(x)==3 else embarkMap[x] 
                               if row.index(x)==10 else x,row)
                
                # note that the test csv does not contain a column with the labels (Survived (1/0))
                # We'll have to deal with blanks a little differently here
                # We can't just ignore all the passengers with missing values as the Kaggle 
                # submission needs you to predict the outcome even for passengers whose attribute data
                # is missing
                # So, for now let's use a default value whenever we encounter a blank
                #featureVector=map(lambda x:0 if x=="" else x, featureVector)
                
                # The second column is Passenger class, let the default value be 2nd class
                if allFeatures[1]=="":
                    allFeatures[1]=2
                # Let the default age be 30
                if allFeatures[4]=="":
                    allFeatures[4]=30
                # Let the default number of companions be 0 (assume if we have no info, the passenger
                # was travelling alone)
                if allFeatures[5]=="":
                    allFeatures[5]=0
                # By eyeballing the data , the average fare seems to be around 30
                if allFeatures[8]=="":
                    allFeatures[8]=32
                featureVector=[allFeatures[header.index(feature)] for feature in features]     
                transformData.append(featureVector)
                ids.append(row[0])
            lineNum=lineNum+1 
    return transformData,ids


# In[53]:

# Let's get a new result with this new function 
resultFile="/Users/swethakolalapudi/Desktop/Kaggle/Titanic/result3.csv"
titanicTest(clf,resultFile,transformTestDataTitanicv2)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



