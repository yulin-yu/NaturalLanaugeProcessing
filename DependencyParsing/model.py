import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ParserModel(nn.Module):


    def __init__(self, config, word_embeddings=None, pos_embeddings=None,
                 dep_embeddings=None):
        super(ParserModel, self).__init__()

        self.config = config
        
        # These are the hyper-parameters for choosing how many embeddings to
        # encode in the input layer.  See the last paragraph of 3.1
        n_w = config.word_features_types # 18
        n_p = config.pos_features_types # 18
        n_d = config.dep_features_types # 12

        # Copy the Embedding data that we'll be using in the model.  Note that the
        # model gets these in the constructor so that the embeddings can come
        # from anywhere (the model is agnostic to the source of the embeddings).
        self.word_embeddings =  word_embeddings 
        self.pos_embeddings = pos_embeddings # TODO input, output,### hiding lever, hiding 先linear再3次方 , output layer 直接
        self.dep_embeddings = dep_embeddings
        
        # Create the first layer of the network that transform the input data
        # (consisting of embeddings of words, their corresponding POS tags, and
        # the arc labels) to the hidden layer raw outputs.

        # TODO #arch label, 三个label整合到一个array，如果33不需要空着 
        context_size = n_w+n_p+n_d
        self.hidden = nn.Linear(context_size * config.embedding_dim, config.l1_hidden_size)
        
        # After the activation of the hidden layer, you'll be randomly zero-ing
        # out a percentage of the activations, which is a process known as
        # "Dropout".  Dropout helps the model avoid looking at the activation of
        # one particular neuron and be more robust.  (In essence, dropout is
        # turning the one network into an *ensemble* of networks).  Create a
        # Dropout layer here that we'll use later in the forward() call.

        # TODO。     # hider layer
        self.dropout = nn.Dropout(p = config.keep_prob)
                
        # Create the output layer that maps the activation of the hidden layer to
        # the output classes (i.e., the valid transitions)

        # TODO outpu tlaery #这里要有一个linear layer
        self.output = nn.Linear(config.l1_hidden_size, config.num_classes)

        # Initialize the weights of both layers
        self.init_weights()
        
    def init_weights(self):
        # initialize each layer's weights to be uniformly distributed within this
        # range of +/-initrange.  This initialization ensures the weights have something to
        # start with for computing gradient descent and generally leads to
        # faster convergence.
        initrange = 0.1
        self.hidden.weight.data.uniform_(-initrange,initrange)
        self.hidden.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange,initrange)
        self.output.bias.data.zero_()



        
    def lookup_embeddings(self, word_indices, pos_indices, dep_indices, keep_pos = 1):
        
        # Based on the IDs, look up the embeddings for each thing we need.  Note
        # that the indices are a list of which embeddings need to be returned.

        # TODO 如何用embedding layer 算embeeding。 # indices 474 choose of Ws 要哪些词 应该不需要找那个词。。。把词拿出来，pos拿出来，depend拿出来，放到embeeding layer里，
        

        w_embeddings = self.word_embeddings(torch.LongTensor(word_indices))
        p_embeddings = self.pos_embeddings(torch.LongTensor(pos_indices))
        d_embeddings = self.dep_embeddings(torch.LongTensor(dep_indices))

        return w_embeddings, p_embeddings, d_embeddings

    def forward(self, word_indices, pos_indices, dep_indices):
        """
        Computes the next transition step (shift, reduce-left, reduce-right)
        based on the current state of the input.
        

        The indices here represent the words/pos/dependencies in the current
        context, which we'll need to turn into vectors.
        """
        
        # Look up the embeddings for this prediction.  Note that word_indices is
        # the list of certain words currently on the stack and buffer, rather
        # than a single word

        # TODO 靠60函数把embeeding算出来
        w_embeddings, p_embeddings, d_embeddings = self.lookup_embeddings(word_indices, pos_indices, dep_indices)
        # Since we're converting lists of indices, we're getting a matrix back
        # out (each index becomes a vector).  We need to turn these into
        # single-dimensional vector (Flatten each of the embeddings into a
        # single dimension).  Note that the first dimension is the batch.  For
        # example, if we have a batch size of 2, 3 words per context, and 5
        # dimensions per embedding, word_embeddings should be tensor with size
        # (2,3,5).  We need it to be a tensor with size (2,15), which makes the
        # input just like that flat input vector you see in the network diagram.
        #
        # HINT: you don't need to copy data here, only reshape the tensor.
        # Functions like "view" (similar to numpy's "reshape" function will be
        # useful here.        

        # TODO 三个embeding word pos depend 整合到一个 vector里去
        #change_stru = ((w_embeddings, p_embeddings, d_embeddings),1).view(-1,config.embedding_dim*(18+18+12))
        c_size = self.config.word_features_types+ self.config.pos_features_types + self.config.dep_features_types 
        combine_3 = torch.cat((w_embeddings, p_embeddings, d_embeddings),1).view(-1,self.config.embedding_dim*(c_size))
        
        # Compute the raw hidden layer activations from the concatentated input
        # embeddings.
        #
        # NOTE: if you're attempting the optional parts where you want to
        # compute separate weight matrices for each type of input, you'll need
        # do this step for each one!

        # TODO。  vector 送hiden layer 



        
        hidden_result = self.hidden(combine_3)
        # Compute the cubic activation function here.
        #
        # NOTE: Pytorch doesn't have a cubic activation function in the library

        # TODO   # 乘三次方
        hidden_result_1 = hidden_result.pow(3)

        # Now do dropout for final activations of the first hidden layer

        # TODO。 #hider layer out put出来 也要用一个linear

        hidden_result_2 = self.dropout(hidden_result_1)
        # Multiply the activation of the first hidden layer by the weights of
        # the second hidden layer and pass that through a ReLU non-linearity for
        # the final output activations.
        #
        # NOTE 1: this output does not need to be pushed through a softmax if
        # you're going to evaluate the output using the CrossEntropy loss
        # function, which will compute the softmax intrinsically as a part of
        # its optimization when computing the loss.

        # TODO。 #training过程 随机 可以放47写 也可以放 叫drop out
        output = self.output(hidden_result_2)

        return output  
#Ng, Ritchie. “Feedforward Neural Network with PyTorch¶.” Deep Learning Wizard, www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/?fbclid=IwAR0Lo3eDZzVojUWFLXbuHYEYDotW6_oscPVG9sIk8yeiDjImsOPReFFve2U.
#akjindal53244. “akjindal53244/dependency_parsing_tf.” GitHub, github.com/akjindal53244/dependency_parsing_tf/blob/master/parser_model.py?fbclid=IwAR1cLCqXQwM3f1T3r0hMXyaFKe2O9rGYVlg2V9bi-ZDO3RqpbRxPqarp5o0.  

