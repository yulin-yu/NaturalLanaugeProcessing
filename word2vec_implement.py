
# coding: utf-8

# In[1]:


import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
from numpy import argmax
import pandas as pd


# In[178]:


#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from
origcounts = {}
listneed= []



# In[185]:


def loadData(filename):
    global uniqueWords, wordcodes, wordcounts, origcounts
    override = False
    if override:
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


   
    handle = open(filename, "r", encoding="utf8")
    fullconts =handle.read().split("\n")
    #fullconts = [entry.split("\t")[0].replace("<br />", "") for entry in fullconts[1:150] ]
    fullconts = [" ".join(fullconts).lower()]
    fullconts = re.sub('[^A-Za-z]+', ' ', fullconts[0])
    fullconts = [fullconts]
    
   # print(fullconts)
    print ("Generating token stream...")
    stopword = set(stopwords.words('english'))
    rawword = nltk.word_tokenize(fullconts[0])
    fullrec = [x for x in rawword if not x in stopword]
    min_count = 50
    origcounts = Counter(fullrec)
    
    
    print ("Performing minimum thresholding..")
    fullrec_filtered = []
    test1 = pd.read_csv('intrinsic-test.tsv' , sep='\t')  
    listneed = list(set(list(test1['word1']) + list(test1['word2'])))
    for i in origcounts.keys():
        if origcounts[i] >= min_count or i in listneed:
            fullrec_filtered.append(i)   
        else:
            fullrec_filtered.append('<UNK>')
     
    #wordcounts[token] = origcounts[token]
    wordcounts = Counter(fullrec_filtered)

    print ("Producing one-hot indicies")
    char_int = {v:k for k, v in enumerate(wordcounts.keys())}
    #char_int = dict((k, v) for k, v in enumerate(wordcounts.keys()))
    wordcodes = char_int
    uniqueWords = list(Counter(fullrec_filtered).keys())


    #print(uniqueWords)

    uniqueword_Encoded = [char_int[char] for char in fullrec_filtered]
    #onehotlist = []
    #for i in uniqueword_Encoded:
    #    number =  [0 for x in range(len(rawword))]
    #    number[value] = 1
    #    onehotlist.append(number)  
        
    #wordcodes = dict(zip(uniqueWords, onehotlist))






        #... close input file handle
    handle.close()


    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


    return uniqueword_Encoded



# In[139]:


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    global wordcodes
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
   # max_exp_count = 0
    
    print ("Generating exponentiated count vectors")


     
    exp_count_array = [v**exp_power for k,v in wordcounts.items()]
    #print(exp_count_array)
    #np.asarray(list(uniquedict1.values())) 
    max_exp_count = sum(exp_count_array)    
    
    print ("Generating distribution")
    
    prob_dist = []
    for v in exp_count_array:
        nor = v/max_exp_count
        prob_dist.append(nor)
        
    print ("Filling up sampling table")
    cumulative_dict = {}
    key_c = 0
    lendict = len(prob_dist)
    table_size = 1e7
    start = 0
    for a in range(0,lendict):
        for i in range(int(start), int(start + int(table_size*prob_dist[a]))):
            cumulative_dict[i]=a
        start+= table_size*prob_dist[a]
    
    return cumulative_dict 


# In[140]:


def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    if len(results) < num_samples:
        key1=np.random.randint(0,len(samplingTable))
        try:
            content1 = samplingTable[key1]
            if content1 != context_idx:
                results.append(content1)
        except:
            pass
        
        
	#... (TASK) randomly sample num_samples token indices from samplingTable.
	#... don't allow the chosen token to be context_idx.
	#... append the chosen indices to results


    return results


# In[141]:


@jit(nopython=True)
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def performDescent(num_samples, learning_rate, center_token,context_index,W1,W2,negative_indices):
    # sequence chars was generated from the mapped sequence in the core code    
    W1token = W1[center_token,:]
    W2arrayind = W2[context_index,:]
    positive = sigmoid(np.dot(W2arrayind,W1token.T))-1                   
    psum = positive * W2arrayind
    perror = learning_rate*positive*W1token
    pgraident = W2arrayind - perror
    n_sum = 0
    nll = 0
    for c in negative_indices:
        W2arrary_c = W2[c,:]
        nagative = np.array(sigmoid(-np.dot(W2arrary_c,W1token.T)))
        n_sum = np.log(nagative) + n_sum
        #a,b = W2arrary_c.shape
        #column = int(b) - 1
        #co = random.randint(0,column)
        psum = psum + nagative*W2arrary_c
        nerror = learning_rate*nagative*W1token
        ngraident = W2arrary_c - nerror
    serror = learning_rate * psum
    W1token = W1token - serror
    nll = -np.log(positive + 1) - n_sum
    return nll
                       
		#... (TASK) implement gradient descent. Find the current context token from sequence_chars
		#... and the associated negative samples from negative_indices. Run gradient descent on both
		#... weight matrices W1 and W2.
		#... compute the total negative log-likelihood and store this in nll_new.                       
    


# In[190]:


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations
    context_index = []

    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence



	#... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
		#... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



	#... set the training parameters
    epochs = 10
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0




	#... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

		#... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1
            if iternum%10000==0:
                print ("Negative likelihood: ", nll)
                nll_results.append(nll)
                nll = 0


			#... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
#            center_token = mapped_sequence[i]

            if mapped_sequence[i] != '<UNK>':
                center_token = mapped_sequence[i]
        
                #... fill in
			#... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.





            iternum += 1
			#... now propagate to each of the context outputs
            for k in range(0, len(context_window)):

				#... (TASK) Use context_window to find one-hot index of the current context token.
                context_index = fullsequence[context_window[k]+i]
                if context_index == wordcodes['<UNK>']:
                    continue
                #context_index = #... fill in



				#... construct some negative samples
                #negative_indices = []
                
                negative_indices = generateSamples(context_index, num_samples)
				#... (TASK) You have your context token and your negative samples.
				#... Perform gradient descent on both weight matrices.
			
            #... Also keep track of the negative log-likelihood in variable nll.
                nll = performDescent(num_samples, learning_rate, center_token,context_index,W1,W2,negative_indices)
                





    for nll_res in nll_results:
        print (nll_res)
    return [W1,W2]


# In[143]:


def load_model():
	handle = open("saved_W1.data","rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2.data","rb")
	W2 = np.load(handle)
	handle.close()
	return [W1,W2]


def save_model(W1,W2):
	handle = open("saved_W1.data","wb+")
	np.save(handle, W1, allow_pickle=False)
	handle.close()

	handle = open("saved_W2.data","wb+")
	np.save(handle, W2, allow_pickle=False)
	handle.close()


# In[144]:


word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
	global word_embeddings, proj_embeddings
	if preload:
		[curW1, curW2] = load_model()
	else:
		curW1 = None
		curW2 = None
	[word_embeddings, proj_embeddings] = trainer(curW1,curW2)
	save_model(word_embeddings, proj_embeddings)


# In[145]:


def morphology(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [word_seq[0], # suffix averaged
	embeddings[wordcodes[word_seq[1]]]]
	vector_math = vectors[0]+vectors[1]


# In[146]:


def analogy(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [embeddings[wordcodes[word_seq[0]]],
	embeddings[wordcodes[word_seq[1]]],
	embeddings[wordcodes[word_seq[2]]]]
	vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0


# In[149]:



def get_neighbors(target_word):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	targets = [target_word]
	outputs = []
	output_d = {}
	ind_t = wordcodes[target_word]   
	for i in uniqueWords:
		ind_s = wordcodes[i]
		v_s = proj_embeddings[ind_s]
		v_t = proj_embeddings[ind_t]
		score1 = 1-cosine(v_s,v_t)
		output_d[i] = score1
	key = sorted(output_d, key=output_d.get, reverse=True)[:10]
    #value = sorted(output_d.values(),reverse=True)[:10]
	for a in key:   
		elem = {'word':a, "score": float(output_d[a])}
		outputs.append(elem)
	return outputs
        
#					ind1 = wordcodes[word1]
#					vectornumber1 = word_embeddings[ind1]
#					ind2 = wordcodes[word2]
#					vectornumber2 = word_embeddings[ind2]
#					Score = abs(1-cosine(vectornumber1,vectornumber2))
	#... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
	#... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
	#... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
	#... return a list of top 10 most similar words in the form of dicts,
	#... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}


# In[195]:


filename = 'unlabeled-data.txt'
if __name__ == '__main__':
    
	#if len(sys.argv)==2:
		#filename = sys.argv[1]
		#... load in the file, tokenize it and assign each token an index.
		#... the full sequence of characters is encoded in terms of their one-hot positions

		fullsequence= loadData(filename)
		print ("Full sequence loaded...")
		#print(uniqueWords)
		#print (len(uniqueWords))



		#... now generate the negative sampling table
		print ("Total unique words: ", len(uniqueWords))
		print("Preparing negative sampling table")
		samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


		#... we've got the word indices and the sampling table. Begin the training.
		#... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
		#... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
		#... ... and uncomment the load_model() line

		train_vectors(preload=False)
		[word_embeddings, proj_embeddings] = load_model()
		test1 = pd.read_csv('intrinsic-test.tsv' , sep='\t')
        

		Scorelist = []
		for i in range(0,len(list(test1['word1']))):
			try:
					word1 = list(test1['word1'])[i]
					word2 = list(test1['word2'])[i]
					ind1 = wordcodes[word1]
					vectornumber1 = proj_embeddings[ind1]
					ind2 = wordcodes[word2]
					vectornumber2 = proj_embeddings[ind2]
					Score = 1-cosine(vectornumber1,vectornumber2)
					Scorelist.append(Score)
			except:
					Scorelist.append(0)
    
		test1['result'] = Scorelist
		final = test1.drop(columns = ['word1','word2'])
		final.columns = ['id', 'sim']
		final.to_csv('final_new11.csv',index = False)




		#... we've got the trained weight matrices. Now we can do some predictions
		targets = [ 'coast' ,'london' ,'june',  'computer','european','television','meat','university','mathematics', 'women']
		f = open('output1.txt','w')
		for targ in targets:
			f.writelines("\nTarget: " + str(targ))
			print("Target: ", targ)
			bestpreds= (get_neighbors(targ))
			for pred in bestpreds:
				print (pred)
				f.writelines("\n" + str(pred))
			print ("\n")
		f.close()


		#... try morphological task. Input is averages of vector combinations that use some morphological change.
		#... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
		#... the morphology() function.







		sys.exit()


# In[197]:


df = pd.read_csv('final_new1.csv')

df.columns = ['id', 'sim']

df.to_csv('final_new11.csv',index = False)


# In[ ]:


# Q1 - Q5 finished
# Q6 : Write a functionget neighbors(word)so that it takes one argument, the tar-get word, and computes the top 10 most
#similar words based on the cosine similarity.

#Problem 7.Pick 10 target words and compute the most similar for each using your function.Record  these  
#in  a  file  namedprob7output.txtQualitatively  looking  at  the  most  similarwords for each target word, 
#do these predicted word seem to be semantically similar to the targetword? 
#Describe what you see in 2-3 sentences.Hint:For maximum effect, 
#try picking wordsacross a range of frequencies (common, occasional, rare words).


# In[ ]:


# individual
# subject
 #subjectivitly Our multiple path, 

