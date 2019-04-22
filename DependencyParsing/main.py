import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from general_utils import get_minibatches
from test_functions import get_UAS, compute_dependencies
from feature_extraction import load_datasets, DataConfig, Flags, punc_pos, pos_prefix
import argparse

import sys
from model import ParserModel
from gensim.models import KeyedVectors



def load_embeddings(config, emb_type='new', emb_file_name=None):
    if emb_type == 'new':
        print('Creating new trainable embeddings')
        word_embeddings = nn.Embedding(config.word_vocab_size, config.embedding_dim)
        pos_embeddings = nn.Embedding(config.pos_vocab_size, config.embedding_dim)
        dep_embeddings = nn.Embedding(config.dep_vocab_size, config.embedding_dim)
    elif emb_type == 'twitter':
        print('Creating new twitterembeddings')
        load = KeyedVectors.load_word2vec_format('twitter.word2vec.100-iter.bin', binary=True, encoding='utf-8', unicode_errors='ignore')   
        pos_embeddings = nn.Embedding(config.pos_vocab_size, config.embedding_dim)
        dep_embeddings = nn.Embedding(config.dep_vocab_size, config.embedding_dim)
        weight = torch.FloatTensor(load.vectors)
        word_embeddings = nn.Embedding.from_pretrained(weight)

#https://pytorch.org/docs/stable/nn.html

        # TODO

    elif emb_type == 'wiki' or emb_type == 'wikipedia':
        # TODO
        print('Creating new wikiembeddings')
        load1 = KeyedVectors.load_word2vec_format('wiki.word2vec.min-100.bin', binary=False, encoding='utf-8', unicode_errors='ignore')   
        pos_embeddings = nn.Embedding(config.pos_vocab_size, config.embedding_dim)
        dep_embeddings = nn.Embedding(config.dep_vocab_size, config.embedding_dim)
        weight = torch.FloatTensor(load1.vectors)
        word_embeddings = nn.Embedding.from_pretrained(weight)
    else:
        raise Error('unknown embedding type!: "%s"' % emb_type)

    return word_embeddings, pos_embeddings, dep_embeddings


def train(save_dir='saved_weights', parser_name='parser', num_epochs=5, max_iters=-1,
          print_every_iters=10):
    """
    Trains the model.

    parser_name is the string prefix used for the filename where the parser is
    saved after every epoch    
    """
    
    # load dataset
    load_existing_dump=False
    print('Loading dataset for training')
    dataset = load_datasets(load_existing_dump)
    # HINT: Look in the ModelConfig class for the model's hyperparameters
    config = dataset.model_config

    print('Loading embeddings')
    word_embeddings, pos_embeddings, dep_embeddings = load_embeddings(config)
    #word_embeddings, pos_embeddings, dep_embeddings = load_embeddings(config, emb_type == 'twitter')  # for twitter
    #word_embeddings, pos_embeddings, dep_embeddings = load_embeddings(config,emb_type == 'wiki') for wiki
    # TODO: For Task 3, add Twitter and Wikipedia embeddings (do this last)    # 把词变vector
    
    if False:
        # Switch to True if you want to print examples of feature types
        print('words: ',len(dataset.word2idx))
        print('examples: ',[(k,v) for i,(k,v) in enumerate(dataset.word2idx.items()) if i<30])    
        print('\n')
        print('POS-tags: ',len(dataset.pos2idx))
        print(dataset.pos2idx)
        print('\n')
        print('dependencies: ',len(dataset.dep2idx))
        print(dataset.dep2idx)
        print('\n')
        print("some hyperparameters")
        print(vars(config))
    
    # load parser object
    parser = ParserModel(config, word_embeddings, pos_embeddings, dep_embeddings)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser.to(device)

    # set save_dir for model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create object for loss function。  # 提示：cross information entropy # entropy loss   (挑最大的做prediction )
    loss_fn = nn.CrossEntropyLoss()
    # TODO
    # citation: https://pytorch.org/docs/stable/nn.html
    # create object for an optimizer that updated the weights of our parser     #adam 算法     #potruch 里有      #
    # model.  Be sure to set the learning rate based on the parameters!
    params = parser.parameters()
    optimizer = torch.optim.Adam(params, lr=config.lr)
    
    for epoch in range(1, num_epochs+1):
        
        ###### Training #####

        # load training set in minibatches
        for i, (train_x, train_y) in enumerate(get_minibatches([dataset.train_inputs,
                                                                dataset.train_targets], \
                                                               config.batch_size,
                                                               is_multi_feature_input=True)):

            word_inputs_batch, pos_inputs_batch, dep_inputs_batch = train_x

            #torch_ex_float_tensor = torch.from_numpy(numpy_ex_array)


            # Convert the numpy data to pytorch's tensor represetation.  They're              #word impute
            # numpy objects initially.  NOTE: In general, when using Pytorch,
            # you want to send them to the device that will do th e computation
            # (either a GPU or CPU).  You do this by saying "obj.to(device)"              #写不写无所谓，教的时候写上
            # where we've already created the device for you (see above where we
            # did this for the parser).  This ensures your data is running on
            # the processor you expect it to!
            word_inputs_batch = torch.from_numpy(np.array(word_inputs_batch)).to(device)
            pos_inputs_batch = torch.from_numpy(np.array(pos_inputs_batch)).to(device)         # TODO     #batch数据准备好 一个句子 pascing 列表
            dep_inputs_batch = torch.from_numpy(np.array(dep_inputs_batch)).to(device)

            # Convert the labels from 1-hot vectors to a list of which index was
            # 1, which is what Pytorch expects.  HINT: look for the "argmax"
            # function in numpy.
            labels = np.argmax(train_y,axis =1) # TODO。  #第几列最大的 还是第几列最小 把label拿出来 词-》数-〉probablity

            # Convert the label to pytorch's tensor
            labels = torch.from_numpy(labels) # TODO     #变pytorch数据结构

            # This is just a quick hack so you can cut training short to see how
            # things are working.  In the final model, make sure to use all the data!
            if max_iters >= 0 and i > max_iters:
                break

            # Some debugging information for you
            if i==0 and epoch==1:
                print("size of word inputs: ",word_inputs_batch.size())
                print("size of pos inputs: ",pos_inputs_batch.size())
                print("size of dep inputs: ",dep_inputs_batch.size())
                print("size of labels: ",labels.size())

            #
            #### Backprop & Update weights ####
            #
            
            # Before the backward pass, use the optimizer object to zero all of
            # the gradients for the variables

            # TODO           #现成的trouch写好的 每个todo一行代码 # 训练数据 标准数据 告诉他 算偏差 去优化model # 让他梯度下降
            
            optimizer.zero_grad()   # my input
            # For the current batch of inputs, run a full forward pass through the
            # data and get the outputs for each item's prediction.
            # These are the raw outputs, which represent the activations for
            # prediction over valid transitions.
            
            outputs = parser(word_inputs_batch, pos_inputs_batch, dep_inputs_batch) # TODO。         

            # Compute the loss for the outputs with the labels.  Note that for
            # your particular loss (cross-entropy) it will compute the softmax
            # for you, so you can safely pass in the raw activations.

            loss = loss_fn(outputs,labels) # TODO

            # Backward pass: compute gradient of the loss with respect to model parameters

            # TODO
            loss.backward()

            # Perform 1 update using the optimizer

            # TODO 
            optimizer.step()

            #https://pytorch.org/docs/stable/optim.html
            # Every 10 batches, print out some reporting so we can see convergence
            if i % print_every_iters == 0:
                print ('Epoch: %d [%d], loss: %1.3f, acc: %1.3f' \
                       % (epoch, i, loss.item(),
                          int((outputs.argmax(1)==labels).sum())/len(labels)))

        print("End of epoch")

        # save model
        save_file = os.path.join(save_dir, '%s-epoch-%d.mdl' % (parser_name, epoch))
        print('Saving current state of model to %s' % save_file)
        torch.save(parser, save_file)

        ###### Validation #####
        print('Evaluating on valudation data after epoch %d' % epoch)
        
        # Once we're in test/validation time, we need to indicate that we are in
        # "evaluation" mode.  This will turn off things like Dropout so that
        # we're not randomly zero-ing out weights when it might hurt performance
        parser.eval()

        # Compute the current model's UAS score on the validation (development)
        # dataset.  Note that we can use this held-out data to tune the
        # hyper-parameters of the model but we should never look at the test
        # data until we want to report the very final result.
        compute_dependencies(parser, device, dataset.valid_data, dataset)
        valid_UAS = get_UAS(dataset.valid_data)
        print ("- validation UAS: {:.2f}".format(valid_UAS * 100.0))

        # Once we're done with test/validation, we need to indicate that we are back in
        # "train" mode.  This will turn back on things like Dropout
        parser.train()

    return parser
        
def test(parser):
    
    # load dataset
    load_existing_dump=False
    print('Loading data for testing')
    dataset = load_datasets(load_existing_dump)
    config = dataset.model_config
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Make sure the parser is in evaluation mode so it's not using things like dropout
    parser.eval()

    # Compute UAS (unlabeled attachment score), which is the standard evaluate metric for parsers.
    #
    # For details see
    # http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002
    # Chapter 6.1
    compute_dependencies(parser, device, dataset.test_data, dataset)
    valid_UAS = get_UAS(dataset.test_data)
    print ("- test UAS: {:.2f}".format(valid_UAS * 100.0))



if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_train_iters", help="Maximum training " +
                           "iterations during one epoch (debug only)",
                           type=int, default=-1, required=False)
    argparser.add_argument("--parser_name", help="Name used to save parser",
                           type=str, default="parser", required=False)    
    argparser.add_argument("--num_epochs", help="Number of epochs",
                           type=int, default=10, required=False)
    argparser.add_argument("--print_every_iters", help="How often to print "
                           + "updates during training",
                           type=int, default=50, required=False)
    argparser.add_argument("--train", help="Train the model",
                           action='store_true')
    argparser.add_argument("--test", help="Test the model",
                           action='store_true')    
    argparser.add_argument("--load_model_file", help="Load the specified "
                           + "saved model for testing",
                           type=str, default=None)    
    
    args = argparser.parse_args()    
    parser = None
    if args.train:
        parser = train(max_iters=args.max_train_iters, num_epochs=args.num_epochs,
                       parser_name=args.parser_name,
                       print_every_iters=args.print_every_iters)
    if args.test:
        if parser is None or args.load_model_file is not None:
            # load parser object
            print('Loading saved parser for testing')
            load_file = args.load_model_file

            if load_file is None:
                # Back off to see if we can keep going
                load_file = 'saved_weights/parser-epoch-1.mdl'

            print('Testing using model saved at %s' % load_file)
            parser = torch.load(load_file)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            parser.to(device)

        test(parser)

    if not (args.train or args.test):
        print('Neither --train nor --test specified! Doing nothing...')


#Citation: 
#“Word Embeddings: Encoding Lexical Semantics¶.” Word Embeddings: Encoding Lexical Semantics - PyTorch Tutorials 1.0.0.dev20190327 Documentation, pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py.
# Tatsuokun. “Tatsuokun/Deepdep.” GitHub, github.com/tatsuokun/deepdep/blob/master/DeNSe/main.py?fbclid=IwAR1QuTCr1FEIRy-u15kkMSczN3FcVd1_5muV59ZnWtKm3SW3_n2wv7lICu0.
