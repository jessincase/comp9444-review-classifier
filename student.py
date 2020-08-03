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
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe
import tensorflow as tf
# import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # negative sampling

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    
    return batch

stopWords = {}
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

    return datasetLabel.sub_(1)

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    netOutput = netOutput.argmax(dim=1, keepdim=True)

    return netOutput.sub_(-1)

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
        class_num = 5
        filters_num = 100
        kernel_sizes = [3,4,5]
        dropout = 0.5 
        embed_dim = 300
        
        self.convs = tnn.ModuleList([
            tnn.Conv2d(1, filters_num, (size, embed_dim)) 
            for size in kernel_sizes
        ])
        self.dropout = tnn.Dropout(dropout)
        self.fc = tnn.Linear(len(kernel_sizes) * filters_num, class_num)


    def forward(self, x, length):
        # [(W−K+2P)/S]+1.
        
        # padding = tnn.ZeroPad2d((0, 0, 0, 125-x.size(1)))
        x = x.unsqueeze(1)
        # x = padding(x)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size, kernel_num, h)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(batch_size, kernel_num)]
        x = torch.cat(x, 1) # (batch_size, kernel_num * len(kernel_sizes)) 
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
        

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        return F.cross_entropy(output, target.type(torch.LongTensor))
        #return F.nll_loss(output, target.type(torch.LongTensor))

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
# optimiser = toptim.Adam(params=net.parameters(),
#                                      lr=1e-4,
#                                      weight_decay=1e-5)
optimiser = toptim.SGD(net.parameters(), lr=0.01, momentum=0.01)
