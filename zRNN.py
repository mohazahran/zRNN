'''
Created on Feb 21, 2018

@author: mohame11
'''
import numpy as np
import nltk
import csv
import itertools
import math
import random
from scipy import special
import pickle as pkl
import re

def sigmoid(x):
    #return 1. / (1. + np.exp(-x))
    return special.expit(x)

class zLSTM(object):
    '''
    classdocs
    '''
    def __init__(self, inputDim=10, hiddenDim=10, learningRate = 0.01, clipGradients = True, useAdaGrad = True, batchSize = 10):
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.clipGradients = clipGradients
        self.useAdaGrad = useAdaGrad
        self.batchSize = batchSize
        
        self.Wxh = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        self.Whh = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        self.Why = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        
        self.bh = np.random.uniform(-0.1, 0.1, (1,hiddenDim))
        self.by = np.random.uniform(-0.1, 0.1, (1,inputDim))
        
        #memory for gradients for adagrad
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        self.memory={}
        self.memory['dWxh'] = mWxh; self.memory['dWhh'] = mWhh; self.memory['dWhy'] = mWhy
        self.memory['dbh'] = mbh; self.memory['dby'] = mby    
        
    def forwardPass(self, inputSeq): #inputSeq is a sequence of input vectors (e.g. an input sentence)
        inputCount = len(inputSeq) #e.g. number of words in a sentence
        
        # The outputs at each time step.
        H = np.zeros((inputCount, self.hiddenDim))
        Y = np.zeros((inputCount, self.hiddenDim))
        softmaxPredictions = np.zeros((inputCount, self.hiddenDim))
        
        for t in range(inputCount): # for each item in a sequence (e.g. word in a sentence)
            xt = np.zeros(self.inputDim)
            xt[inputSeq[t]] = 1.0
            H[t] = np.tanh( np.dot(self.Wxh, xt) + np.dot(self.Whh, H[t-1]) + self.bh ) # hidden state
            Y[t] = np.dot(self.Why, H[t]) + self.by # unnormalized log probabilities for next chars
            softmaxPredictions[t] = self.stable_softmax(Y[t])
            
            
        
        return softmaxPredictions, Y, H
    
    def generate(self, xt, ht_1, ct_1):
        zt_ = np.dot(self.Wz, xt) + np.dot(self.Rz, ht_1) + self.bz
        zt = np.tanh(zt_)
        
        it_ = np.dot(self.Wi, xt) + np.dot(self.Ri, ht_1) + self.bi
        it = sigmoid(it_)
        
        ot_ = np.dot(self.Wo, xt) + np.dot(self.Ro, ht_1) + self.bo
        ot = sigmoid(ot_)
        
        ft_ = np.dot(self.Wf, xt) + np.dot(self.Rf, ht_1) + self.bf
        ft = sigmoid(ft_)        
        
        ct = zt * it + ct_1 * ft
        
        ht = np.tanh(ct) * ot
        
        softmaxPredictions = self.stable_softmax(ht)
        
        return softmaxPredictions, ht[0], ct[0]
    
    def calculate_loss_batch(self, inputBatch, trueOutputBatch):
        loss = 0.0
        for i in range(len(inputBatch)):
            loss += self.calculate_loss(inputBatch[i], trueOutputBatch[i])
            
        return loss / len(inputBatch)
                    
    def calculate_loss(self, inputSeq, trueOutputSeq):
        L = 0
        softmaxPredictions, Y, H = self.forwardPass(inputSeq)
        for t in range(len(H)):
            L -= np.log(softmaxPredictions[t][trueOutputSeq[t]])
        
        return L

    
    def backProp(self, inputBatch, trueOutputBatch):
        deltas = {}
        dWxh = np.zeros(self.Wxh.shape)
        dWhh = np.zeros(self.Whh.shape)
        dWhy = np.zeros(self.Why.shape)
        
        dbh = np.zeros(self.bh.shape)
        dby = np.zeros(self.by.shape)
        
        for b in range(len(inputBatch)): #for each input sentence in the batch
            X = inputBatch[b]
            
            inputCount = len(X)
            softmaxPredictions, Y, H = self.forwardPass(X)
            
            T = len(H)
            
            dH = np.zeros((inputCount, self.hiddenDim))
            
            dhnext = np.zeros((H[0].shape[0], 1))
            
            for t in reversed(range(T)): #from T down to 0
                xt = np.zeros(self.inputDim)
                xt[X[t]] = 1.0
                xt = xt.reshape(xt.shape[0],1)
                
                Y = np.zeros(self.inputDim)
                Y[trueOutputBatch[b][t]] = 1.0
                
                
                dH[t] += softmaxPredictions[t] - Y
                ht = H[t].reshape(H[t].shape[0],1)
                dht = dH[t].reshape(dH[t].shape[0],1)
                if t-1 < 0:
                    ht_1 = np.zeros((self.inputDim,1))
                else:
                    ht_1 = H[t-1].reshape(H[t-1].shape[0],1)
                
                #H[t] -> hs
                #dH[t] - >dy
                #dWhy += np.dot(dy, hs[t].T)
                dWhy += np.dot(dht, ht.T)
                dby += dht.T
                
                #dh = np.dot(Why.T, dy) + dhnext # backprop into h
                dh = np.dot(self.Why.T, dht) + dhnext # backprop into h
                
                #dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
                dhraw = (1 - ht * ht) * dh # backprop through tanh nonlinearity
                dbh += dhraw.T
                dWxh += np.dot(dhraw, xt.T)
                dWhh += np.dot(dhraw, ht_1.T)
                dhnext = np.dot(self.Whh.T, dhraw)                        
        
        deltas['Wxh'] = dWxh; deltas['Whh'] = dWhh; deltas['Why'] = dWhy
        deltas['bh'] = dbh; deltas['by'] = dby
        return deltas
   
    
    def SGD(self, deltas):
        #clip gradients
        for d in deltas:
            if self.clipGradients:
                np.clip(deltas[d], -5, 5, out=deltas[d]) # clip to overcome exploding gradients
            if self.useAdaGrad:
                self.memory['d'+d] += deltas[d] * deltas[d] # updating memory
        
        
            
        #do SGD update
        if self.useAdaGrad:
            self.Wxh = self.Wxh - self.learningRate * deltas['Wxh'] / np.sqrt(self.memory['dWxh'] + 1e-8)
            self.Whh = self.Whh - self.learningRate * deltas['Whh'] / np.sqrt(self.memory['dWhh'] + 1e-8)
            self.Why = self.Why - self.learningRate * deltas['Why'] / np.sqrt(self.memory['dWhy'] + 1e-8)
            
            self.bh = self.bh - self.learningRate * deltas['bh'] / np.sqrt(self.memory['dbh'] + 1e-8)
            self.by = self.by - self.learningRate * deltas['by'] / np.sqrt(self.memory['dby'] + 1e-8)
        else:
            self.Wxh = self.Wxh - self.learningRate * deltas['Wxh'] 
            self.Whh = self.Whh - self.learningRate * deltas['Whh'] 
            self.Why = self.Why - self.learningRate * deltas['Why'] 
            
            self.bh = self.bh - self.learningRate * deltas['bh'] 
            self.by = self.by - self.learningRate * deltas['by'] 
    

    def stable_softmax(self, X):
        #table softmax
        #exps = np.exp(X - np.max(X))
        #return exps / np.sum(exps)
    
        exps = np.exp(X)
        return exps / np.sum(exps)


    def train(self, trainingSet = [], trainingTruth = []):
        batchCount = int(math.ceil(float(len(trainingSet)) / self.batchSize))
        for b in range(batchCount):
            X_batch = trainingSet[b*self.batchSize : (b+1)*self.batchSize]
            Y_batch = trainingTruth[b*self.batchSize : (b+1)*self.batchSize]
            
            deltas = self.backProp(X_batch, Y_batch)
            self.SGD(deltas)
            
    
    
def clean_Data(myData):
    cleanedData = []   
    p1 = re.compile("\w+-\w+")   
    p2 = re.compile("\W\d+[A-Za-z]+")
    p3 = re.compile("[A-Za-z]+\d+\W")
    p4 = re.compile("[^A-Za-z0-9\s']") #remove special chars
    p5 = re.compile("[^A-Za-z](\d+\s*)+") #consectutive NUMs
    for line in myData:    
        line = line.strip()    
        matches = p1.findall(line)
        for m in matches:
            line = line.replace(m, m.replace('-',''))
        matches = p2.findall(line)
        for m in matches:
            id = 1
            for c in m:
                if c.isdigit():
                    id +=1
                elif(id != 1):
                    break
            s = m[:id] + ' ' + m[id:].strip()
            line = line.replace(m, s)
        
        matches = p3.findall(line)
        for m in matches:
            id = 0
            for c in m:
                if c.isdigit():
                    break
                else:
                    id +=1
            s = m[:id] + ' ' + m[id:].strip()
            line = line.replace(m, s)
        
        matches = p4.findall(line)
        for m in matches:
            line = line.replace(m, m.replace(m,' '))
            
        matches = p5.findall(line)
        for m in matches:
            line = line.replace(m, ' num ')
            
                
            
        cleaned = ' '.join(line.split())          
        cleaned = cleaned.lower()
        cleaned = cleaned.strip()
        if len(cleaned) <= 1:
            continue
        cleanedData.append(cleaned)        
    return cleanedData

def preProcessing_charBased():
    f = open('toyExample', 'r') 
    data = f.readlines()
    chars = '\n'.join(clean_Data(data))
    uniqueChars = set(chars)
    c2i = { ch:i for i,ch in enumerate(uniqueChars) }
    i2c = { i:ch for i,ch in enumerate(uniqueChars) }
    
    print 'The number of chars = %d' % len(chars)
    print "The number of unique chars = %d." % len(c2i)
     
    # Create the training data
    X_train = np.asarray([[c2i[c] for c in chars[:-1]]])
    Y_train = np.asarray([[c2i[c] for c in chars[1:]]])
    return X_train, Y_train, c2i, i2c

def preProcessing(vocabSize):
    vocabulary_size = vocabSize
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    f = open('toyExample', 'r')
    # Split full comments into sentences
    sentences = f.readlines()
    sentences = clean_Data(sentences)
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
         
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    
    
     
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())
     
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
     
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
     
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        #tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
        #dropout all unknown words
        tokenized_sentences[i] = [w for w in sent if w in word_to_index]
     
    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
     
    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, Y_train, word_to_index, index_to_word

def generateText(lstm, w2i, i2w, startWordSeed, wordCount):
    if startWordSeed not in w2i:
        print 'Word is not in vocab, try different word!'
        return
    sent = startWordSeed + ','
    genWord = startWordSeed
    ht_1 = np.zeros(lstm.hiddenDim)
    while wordCount > 0:
        xt = np.zeros(lstm.inputDim)
        xt[w2i[genWord]] = 1.0
        ht = np.tanh(np.dot(lstm.Wxh, xt) + np.dot(lstm.Whh, ht_1) + lstm.bh)
        yt = np.dot(lstm.Why, ht[0]) + lstm.by
        softmaxPredictions = lstm.stable_softmax(yt)
        genWordId = softmaxPredictions[0].argmax()
        genWord = i2w[genWordId]
        sent += genWord + ','
        ht_1 = ht[0]
        wordCount -= 1
    return sent
    
    

def main():
    np.random.seed(10)
    
    #X_all, Y_all, w2i, i2w = preProcessing(10)
    X_all, Y_all, w2i, i2w = preProcessing_charBased()
    
    D = len(w2i) # Number of input dimension == number of items in vocabulary
    H = D # Number of LSTM layer's neurons
    epochs = 500000
    valQuota = 0.0
    
    #X_all = X_all[:2]
    #Y_all = Y_all[:2]
    valSize = int(valQuota * len(X_all))
    X_val = X_all[:valSize]
    Y_val = Y_all[:valSize]
    X_train = X_all[valSize:]
    Y_train = Y_all[valSize:]
    
    
    lstm = zLSTM(inputDim=D, hiddenDim=H, learningRate=0.01, clipGradients= True, useAdaGrad=True, batchSize = 1)
    
    print 'Starting training'
    
    crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
    print 'Cross Entropy TRAIN Loss = ', crossEntropyLoss
    
    if valSize:
        crossEntropyLoss = lstm.calculate_loss_batch(X_val, Y_val)
        print 'Cross Entropy VAL Loss   = ', crossEntropyLoss
    for i in range(epochs):
        #print 'Epoch#',i
        
        #lstm.train(X_train, Y_train, batchSize=10, learningRate=lr/float(i+1))
        lstm.train(X_train, Y_train)
        
        if i % 100 == 0:
            print 'Epoch#',i
            crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
            print 'Cross Entropy TRAIN Loss = ', crossEntropyLoss
            
            if valSize:
                crossEntropyLoss = lstm.calculate_loss_batch(X_val, Y_val)
                print 'Cross Entropy VAL Loss   = ', crossEntropyLoss
            
            print 'Generated chars:\n', generateText(lstm, w2i, i2w, ' ', 100)
            #pkl.dump([lstm,w2i,i2w], open('lstm.pkl', 'wb'))
            #print 'Model is saved to lstm.pkl'
            
        #shuffle the training set for every epoch
        #combined = list(zip(X_train, Y_train))
        #random.shuffle(combined)
        #X_train[:], Y_train[:] = zip(*combined)
        
    pkl.dump([lstm,w2i,i2w], open('lstm.pkl', 'wb'))
    print 'Model is saved to lstm.pkl'
    
    

def simulateData():
    lst = pkl.load(open('lstm.pkl', 'rb'))
    lstm = lst[0]
    w2i = lst[1]
    i2w = lst[2]
    sent = generateText(lstm, w2i, i2w, ' ', 500)
    print sent
    

if __name__ == '__main__':
    #simulateData()
    main()