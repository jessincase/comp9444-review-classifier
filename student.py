#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

# Stop words are words which we want to remove from our input reviews
stopWords = {}
# For example, these are from the python NLTK library
'''
stopWords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
             'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
             'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
             't', 'can', 'will', 'just', 'don', 'should', 'now'}
'''

wordVectors = GloVe(name='6B', dim=300)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    # Convert to longTensor (i.e. int)
    convertedLabel = datasetLabel.type(torch.LongTensor)
    # Subtract 1 for classifier target values 0-4
    convertedLabel = torch.add(convertedLabel, -1)
    # Create Tensor as one-hot encoding (makes it easier to do MSE)
    oneHot = torch.zeros(convertedLabel.shape[0], 5)
    for index in range(oneHot.shape[0]):
        oneHot[index, convertedLabel[index]] = 1
    #return convertedLabel # goes 0 to 4
    return oneHot # one hot encoding
    #return datasetLabel # goes 1 to 5

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # Find index of max response, gives values 0-4
    netOutput = netOutput.argmax(dim=1, keepdim=True)
    # Add one to get back to 1-5 range
    netOutput = torch.add(netOutput, 1)
    # Convert to float
    netOutput = netOutput.float()
    return netOutput

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()

    def forward(self, input, length):
        pass

# Basic LSTM
class LSTMBasedNetwork(tnn.Module):
    def __init__(self):
        super(LSTMBasedNetwork, self).__init__()
        # Define parameters
        # Input size is equivalent to the GloVe dimension
        self.input_size = 300
        self.num_layers = 2
        self.hidden_size = 128
        self.num_classes = 5
        # Define network components
        self.lstm = tnn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True)
        self.fully_connected = tnn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input, length):
        # Input is of shape (batchSize, maxLengthInBatch, wordEmbeddingDim)
        # length provides the 'true' length of the sentence
        # Initialise the hidden and cell state
        h0 = torch.zeros(self.num_layers, input.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.shape[0], self.hidden_size).to(device)
        # Run the model
        output, _ = self.lstm(input, (h0, c0))
        # Take the hidden state based on the true length of the sentences
        # Initialise tensor with zeros
        state_out = torch.zeros(input.shape[0], self.hidden_size)
        # Loop through output of lstm and take the hidden state corresponding to the true length
        for index in range(input.shape[0]):
            state_out[index, :] = output[index, length[index] -1,:]
        # Pass hidden states to fully connected layer
        output = self.fully_connected(state_out)
        # Take log(softmax) of the output
        output = tnn.functional.log_softmax(output, dim=1)
        return output

# Bi-directional LSTM
class BRNN(tnn.Module):
    def __init__(self):
        super(BRNN, self).__init__()
        # Define parameters
        # Input size is equivalent to the GloVe dimension
        self.input_size = 300
        self.num_layers = 2
        self.hidden_size = 128
        self.num_classes = 5
        # Define network components
        self.lstm = tnn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True)
        self.fully_connected = tnn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, input, length):
        # Input is of shape (batchSize, maxLengthInBatch, wordEmbeddingDim)
        # length provides the 'true' length of the sentence
        # Initialise the hidden and cell state
        h0 = torch.zeros(self.num_layers*2, input.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, input.shape[0], self.hidden_size).to(device)
        # Run the model
        output, _ = self.lstm(input, (h0, c0))
        # Take the hidden state based on the true length of the sentences
        # Initialise tensor with zeros
        state_out = torch.zeros(input.shape[0], self.hidden_size*2)
        # Loop through output of lstm and take the hidden state corresponding to the true length
        for index in range(input.shape[0]):
            state_out[index, :] = output[index, length[index] -1,:]
        # Pass hidden states to fully connected layer
        output = self.fully_connected(state_out)
        # Take log(softmax) of the output
        #output = tnn.functional.log_softmax(output, dim=1)
        return output

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.myloss = tnn.CrossEntropyLoss()
        self.myMSEloss = tnn.MSELoss()

    def forward(self, output, target):
        val = self.myMSEloss(output, target)
        return val

class DirectMSEloss(tnn.Module):

    def __init__(self):
        super(DirectMSEloss, self).__init__()

    def forward(self, output, target):
        #print("Input ===")
        #print(output)
        #print("Target ===")
        #print(target)
        # Convert from one-hot back to labels (I know this is redundant)
        y = target.argmax(dim=1, keepdim=True)
        # Add one to get back to 1-5 range
        y = torch.add(y, 1)

        # Do same for predictions
        x = output.argmax(dim=1, keepdim=True)
        # Add one to get back to 1-5 range
        x = torch.add(x, 1)

        #print("x ===== ")
        #print(x)
        #print("y =====")
        #print(y)
        # Calculate the MSE
        losses = torch.zeros(x.shape[0], requires_grad=True)
        with torch.no_grad():
            for i in range(x.shape[0]):
                losses[i] = (x[i] - y[i])**2
        #print("loss is = " + str(torch.mean(losses)))
        return torch.mean(losses)

class DirectMSEloss(tnn.Module):

    def __init__(self):
        super(DirectMSEloss, self).__init__()

    def forward(self, output, target):
        #print("Input ===")
        #print(output)
        #print("Target ===")
        #print(target)
        # Convert from one-hot back to labels (I know this is redundant)
        y = target.argmax(dim=1, keepdim=True)
        # Add one to get back to 1-5 range
        y = torch.add(y, 1)

        # Do same for predictions
        x = output.argmax(dim=1, keepdim=True)
        # Add one to get back to 1-5 range
        x = torch.add(x, 1)

        #print("x ===== ")
        #print(x)
        #print("y =====")
        #print(y)
        # Calculate the MSE
        losses = torch.zeros(x.shape[0], requires_grad=True)
        with torch.no_grad():
            for i in range(x.shape[0]):
                losses[i] = (x[i] - y[i])**2
        #print("loss is = " + str(torch.mean(losses)))
        return torch.mean(losses)

# Define the network to be used
net = LSTMBasedNetwork()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = DirectMSEloss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.5
batchSize = 64
epochs = 10
# Use optimiser
lr = 4
mom = 13
# optimiser = toptim.Adam(net.parameters(),lr=lr)
optimiser = toptim.SGD(net.parameters(), lr=lr, momentum=mom)