
# coding: utf-8

# In[2]:

# numpy and scipy are pretty awesome python libraries for scientific computing
# Let's quickly do a round up of what these 2 libraries can help us do

# Python lists are amazing, but many times you'll want to put your data into grids
# and perform mathematical operations with them. A numpy array is a way to put your data
# in a matrix (grid). 
# Numpy then has a number of cool functions to manipulate this data
import numpy as np
import scipy
# An array can have any number of dimensions 
# A 1-d array is like a list (The number of dimensions is also known as rank)
array1d=np.array([0,2,4,6,8])
print array1d


# In[4]:

# A 2-d array is like a 2-d grid. Create it using a list of lists
array2d=np.array([[1,2,3],[4,5,6]])

print array2d


# In[5]:

print type(array1d)


# In[6]:

# shape is a tuple which contains the size of the array ie (the number of rows, the number of 
# columns)
print array1d.shape


# In[7]:

print array2d.shape


# In[8]:

# You can index elements of an array pretty much like you do with lists
# Like with lists, the indexing starts from 0
print array1d[0]

# With a 2d array the syntax is slightly different. If you had a list of lists, and you 
# wanted the element in the ith row and the jth column you would index it by saying 
# listOfLists[i][j]
# In a numpy array, you can index the element in the ith row and jth column as 
# arrayNumpy[i,j]

print array2d[1,2]
# this will print the element in the second row and third column of array2d (indexing starts from
# 0)


# In[9]:

# Just like with lists, you can use : to specify "from the beginning" or "till the end"
print array1d[1:]


# In[10]:

print array2d[1,:1] # This will print the elements in the second row , from the first element in 
# the second row, till the second element (not including the second element)


# In[11]:

print array2d[1,1:] # The elements in the second row, from the second element onwards


# In[12]:

# In all the cases above, the result of the indexing is a numpy array which is a subset
# of the original array 
subarray2d=array2d[1,1:]
print type(subarray2d)


# In[13]:

# There are other cool ways to index numpy arrays 
new2dArray=np.array([[1,2,4,5,9],[3,4,6,6,10],[15,3,2,14,7]])

# Let's say you want the elements with indices [1,2], [0,4], [2,3]

newSubArray=new2dArray[[1,0,2],[2,4,3]] # Arrays of integers to specify the indices
print newSubArray


# In[14]:

# You can also use boolean indexing, ie subset the array to all the values which satisfy
# a certain condition 
new2dArray[new2dArray>10]


# In[15]:

# new2dArray>10 will return an array with the same size and shape as the original array 
# Each element of that array will be a boolean which says whether the corresponding element 
# of the original array satisfies the given condition 
new2dArray>10


# In[16]:

# You can create some standard kinds of arrays using built-in functions in numpy. 
# Ex: Arrays with all 0s, constant arrays, random numbers etc

arrayOfZeros=np.zeros((2,2),dtype='int64') # Creates an array with all zeros. dtype is an optional
# argument, its default value is float
arrayOfOnes=np.ones((1,2)) # note how the size of the array is passed in as a tuple
arraywithConstantValue = np.full(new2dArray.shape,7) # Creates an array with the same
# size and shape as new2dArray, fills it with value 7 for all elements

identityMatrix = np.eye(2)
# Creates a 2x2 identity matrix ie a square grid of numbers with 2 rows and 2 columns
# All the diagonal elements will be 1s and all the non-diagonal elements will be 0s

print arrayOfZeros,"\n",arrayOfOnes,"\n",arraywithConstantValue,"\n",identityMatrix


# In[17]:

# YOu can also fill an array with random numbers 
arrayOfRandomNumbers = np.random.random((2,2)) 
# This will create a 2x2 array with random numbers between the values 0 and 1. 
# To generate random numbers with a specific distribution, check out the 
# numpy documentation
print arrayOfRandomNumbers


# In[18]:

# You can change the size and shape of your array 
transposeArray=np.transpose(new2dArray)

print new2dArray.shape, transposeArray.shape


# In[19]:

# You can reshape array into any shape as long as the total number of elements remains 
# the same
reshapedArray=np.reshape(new2dArray,[1,15])
print reshapedArray


# In[20]:

reshapedArray=np.reshape(new2dArray,[15,1])
print reshapedArray


# In[21]:

# Normal mathematical operators like +,-,/,* can be used to perform element wise 
# operations. Both arrays have to be of the same dimension 

array1=np.array([[1,2,3],[4,5,6]])
array2=np.array([[7,8,9],[3,2,1]])


# In[22]:

array1+array2


# In[23]:

# instead of the math operators you can use functions equivalent to the operators
# +,-,/,* are equivalent to add, subtract, divide,multiply
np.subtract(array1,array2)


# In[24]:

# Broadcasting is a way in which you can add matrices or arrays of different dimensions
array3=[[1,4,7]]

array1+array3 # This will add array3 to each row of array1


# In[25]:

# The rule of thumb for broadcasting is that the arrays need to be able to align 
# along at least 1 dimension. For more details check the documentation


# In[26]:

# One important operation you'll need is to perform an inner product of 2 arrays 
# or multiply 2 matrices. Matrix multiplication is equivalent to taking inner products
# of every row of the first matrix with every column of the second matrix 
# To multiply 2 matrices, you will need the number of columns of the left array
# to be equal to the number of rows of the right array 

np.dot(array1,array2)# wont work because we have both 2x3 arrays


# In[27]:

np.dot(array1,array2.T) # .T is shorthand for using np.transpose()


# In[28]:

# You can add or multiply all the elements along 1 dimension (or axis), This is 
# like compressing the array along that axis
np.sum(array1,axis=0)


# In[30]:

np.sum(array2,axis=1)


# In[31]:

# stack arrays together using vstack or hstack. These functions take in a list/tuple/
# array of arrays and then stack them vertically or horizontally to make a new array
np.vstack((array1,array2))


# In[33]:

np.hstack((array2,array1))


# In[35]:

# Scipy has many modules that help us compute mathematical functions 
# one example is the spatial module, given 2 points represented by 
# np arrays, scipy can help you find the distance between those points
# The distance metric could be any one of a number of options ex: euclidean
# cosine, correlation, hamming etc 
# pdist will compute pairwise distances between the rows in a numpy array 

from scipy.spatial.distance import correlation, cosine, pdist, squareform 

array1=np.array([0,1,0])
array2=np.array([1,0,0])

correlation(array1,array2)


# In[36]:

allPoints=np.vstack([array1,array2])
d=squareform(pdist(allPoints, 'euclidean'))
# The distance metric can be changed to cosine, correlation or any other distance
# the complete list of options is available in scipy documentation 
# This will compute the pairwise distance between all rows of allPoints
# d will be a square matrix with d[i,j] being the Euclidean distance between
# allPoints[i,:] and allPoints[j,:]
print d


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



