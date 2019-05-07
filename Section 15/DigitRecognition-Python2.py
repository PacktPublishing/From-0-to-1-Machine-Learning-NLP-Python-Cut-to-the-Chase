
# coding: utf-8

# In[14]:

# The objective is to train a multi-layered neural network to 
# identify hand written digits. 

# We'll be using a database called MNIST which has about 60000 handwritten 
# digit images with their labels. We'll use these to train a network. Then
# given a new image, the network should be able to classify it as the right digit 
# between 0-9 
import numpy as np
import os 
# Step 1: We'll download the database of images from MNIST website - this is maintained
# by a famous Neural network researcher named Yann Lecun
def load_dataset():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print ("Downloading ",filename)
        import urllib
        urllib.urlretrieve(source+filename,filename)
    # This will download the specified file from Yann Lecun's website and store it 
    # on our local disk
    
    import gzip
    
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Checks if the specified file is already there on our local disk
        # if not it will download the file 
        with gzip.open(filename,'rb') as f:
            # Open the zip file of images 
            data=np.frombuffer(f.read(), np.uint8, offset=16)
            # This is some boilerplate to extract data from the zip file
            # This data has 2 issues : its in the form of a 1 d array
            # We have to take this array and convert it into images 
            # Each image has 28x28 pixels , its a monochrome image ie only 1 channel (if
            # it were full-color it would have 3/4 channels R,G,B etc)
            
            # data is currently a numpy array which we we want to reshape into 
            # an array of 28x28 images 
            data=data.reshape(-1,1,28,28)
            # The first dimension is the number of images , by making this -1
            # The number of images will be inferred from the value of the other dimensions
            # and the length of the input array
            
            # The second dimension is the number of channels - here this is 1
            
            # The third and fourth dimensions are the size of the image 28x28
            
            # its in the form of bytes 
            return data/np.float32(256)
        # This will convert the byte value to a float32 in the range [0,1]
        
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
                # Read the labels which are in a binary file again 
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(),np.uint8,offset=8)
                    # This gives a numpy array of integers, the digit value corresponding 
                    # to the images we got earlier 
        return data
    # We can now download and read the training and test data sets - both the images 
    # and their labels 
    
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    return X_train, y_train, X_test, y_test


# In[15]:

X_train, y_train, X_test, y_test = load_dataset()


# In[18]:

# WE've got our data ready now. If you want to look at one of the images 
# you can use matplotlib 
import matplotlib 
matplotlib.use('TkAgg') # This is just a default setting for matplotlib for how 
# to render images 

import matplotlib.pyplot as plt 
plt.show(plt.imshow(X_train[3][0]))


# In[24]:

# Step 2: We'll set up a neural network with the required number of layers and nodes 
# We'll also tell the network how it has to train itself 

# We are going to use 2 Python packages called Theano and Lasagne 
# Theano is a mathematical package that allows you to define and perform 
# mathematical computations. - like numpy but with high dimensional arrays
# Higher dimensional arrays are often called Tensors - and Theano is a python 
# package to work with them 

# Lasagne is a library that uses Theano heavily and supports building of 
# neural networks. It comes with functions to set up layers , define error functions 
# train neural networks etc 

# Make sure you install the latest versions of Lasagne and Theano - you can find 
# the install command on the corresponding github pages. 

# WE forgot to import the required libraries :) 
import lasagne 

import theano
import theano.tensor as T 

def build_NN(input_var=None):
    # WE are going to create a neural network with 2 hidden layers of 800 nodes each 
    # The output layer will have 10 nodes - the nodes are numbered 0-9 and the output 
    # at each node will be a value between 0-1. The node with the highest value will 
    # be the predicted output
    
    # First we have an input layer - the expected input shape is 
    # 1x28x28 (for 1 image)
    # We will link this input layer to the input_var (which will be the array of images 
    # that we'll pass in later on)
    l_in = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    
    # We'll add a 20% dropout - this means that randomly 20% of the edges between the 
    # inputs and the next layer will be dropped - this is done to avoid overfitting 
    l_in_drop = lasagne.layers.DropoutLayer(l_in,p=0.2)
    
    # Add a layer with 800 nodes. Initially this will be dense/fully-connected
    # ie. every edge possible 
    # will be drawn. 
    l_hid1= lasagne.layers.DenseLayer(l_in_drop,num_units=800,
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      W=lasagne.init.GlorotUniform())
    # This layer has been initialized with some weights. There are some schemes to 
    # initialize the weights so that training will be done faster, Glorot's scheme
    # is one of them 
    
    # We will add a dropout of 50% to the hidden layer 1 
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1,p=0.5)
    
    # Add another layer, it works exactly the same way! 
    l_hid2= lasagne.layers.DenseLayer(l_hid1_drop,num_units=800,
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      W=lasagne.init.GlorotUniform())
    
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2,p=0.5)
    
    
    # Let's now add the final output layer. 
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=10,
                                     nonlinearity = lasagne.nonlinearities.softmax)
    # the output layer has 10 units. Softmax specifies that each of those 
    # outputs is between 0-1 and the max of the those will be the final prediction 
    
    return l_out # We return the last layer, but since all the layers are linked 
# we effectively return the whole network 
    
    
# We've set up the network. Now we have to tell the network how to train itself 
# ie how should it find the values of all the weights it needs to find 

# We'll initialize some empty arrays which will act as placeholders 
# for the training/test data that will be given to the network 

input_var = T.tensor4('inputs') # An empty 4 dimensional array 
target_var = T.ivector('targets') # An empty 1 dimensional integer array to represent
# the labels 

network=build_NN(input_var) # Call the function that initializes the neural network 

# In training we are going to follow the steps below 
 # a. compute an error function 
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
# Categorical cross entropy is one of the standard error functions with 
# classification problems 
loss = loss.mean()

# b. We'll tell the network how to update all its weights based on the 
# value of the error function 
params = lasagne.layers.get_all_params(network, trainable=True) # Current value of all the weights 
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum = 0.9)

# Nesterov momentum is one of the options that Lasagne offers for updating the weights 
# in a training step . This is based on Stochastic Gradient Descent - the idea is simple
# Find the slope of the error function at the current point and move downwards 
# in the direction of that slope 

# We'll use theano to compile a function that is going to represent a 
# single training step ie. compute the error, find the current weights, update the weights
train_fn = theano.function([input_var, target_var], loss, updates=updates)
# calling this function for a certain number of times will train the neural network 




# In[26]:

# Step 3: We'll feed the training data to the neural network 
num_training_steps = 10 # ideally you can train for a few 100 steps

for step in range(num_training_steps):
    train_err = train_fn(X_train, y_train)
    print("Current step is "+ str(step))
    


# In[27]:

# Step 4: We'll check how the output is for 1 image 
# To check the prediction for 1 image we'll need to set up another function 
test_prediction = lasagne.layers.get_output(network)
val_fn = theano.function([input_var],test_prediction)

val_fn([X_test[0]]) # This will apply the function on 1 image, the first one in the test set

# The max value if for the digit 7 


# In[28]:

# Let's check the actual value 
y_test[0]


# In[32]:

# Step 5 : We'll feed a test data set of 10000 images to the trained neural network
# and check it's accuracy 

# We'll set up a function that will take in images and their labels , feed the images 
# to our network and compute it's accuracy 

test_prediction = lasagne.layers.get_output(network,deterministic=True)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var),dtype=theano.config.floatX)
# Checks the index of the max value in each test prediction and matches it agains the actual 
# value 

acc_fn = theano.function([input_var,target_var],test_acc)

acc_fn(X_test,y_test)
# This is pretty poor accuracy - but to improve it you can run the training for more 
# number of steps. You could also divide the training set arbitrarily into smaller datasets
# and run each of the training set on that smaller dataset. This will run faster 
# and take you to the error minimum faster - while avoiding overfitting 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



