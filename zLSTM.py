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
        
        #self.Wz = np.random.uniform(-np.sqrt(1./inputDim), np.sqrt(1./inputDim), (hiddenDim, inputDim))
        self.Wz = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        self.Wi = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        self.Wf = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        self.Wo = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        
        self.Rz = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        self.Ri = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        self.Rf = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        self.Ro = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        
        self.bz = np.random.uniform(-0.1, 0.1, (1,hiddenDim))
        self.bi = np.random.uniform(-0.1, 0.1, (1,hiddenDim))
        #self.bf = np.ones((1,hiddenDim))
        self.bf = np.random.uniform(-0.1, 0.1, (1,hiddenDim)) #better to be initialized to all ones.
        self.bo = np.random.uniform(-0.1, 0.1, (1,hiddenDim))
        
        #memory for gradients for adagrad
        mdWz = np.zeros(self.Wz.shape) ; mdWi = np.zeros(self.Wi.shape); mdWf = np.zeros(self.Wf.shape); mdWo = np.zeros(self.Wo.shape)
        mdRz = np.zeros(self.Rz.shape); mdRi = np.zeros(self.Ri.shape); mdRf = np.zeros(self.Rf.shape); mdRo = np.zeros(self.Ro.shape)
        mdbz = np.zeros(self.bz.shape); mdbi = np.zeros(self.bi.shape); mdbf = np.zeros(self.bf.shape); mdbo = np.zeros(self.bo.shape)
        
        self.memory={}
        self.memory['dWz'] = mdWz; self.memory['dWi'] = mdWi; self.memory['dWf'] = mdWf; self.memory['dWo'] = mdWo
        self.memory['dRz'] = mdRz; self.memory['dRi'] = mdRi; self.memory['dRf'] = mdRf; self.memory['dRo'] = mdRo
        self.memory['dbz'] = mdbz; self.memory['dbi'] = mdbi; self.memory['dbf'] = mdbf; self.memory['dbo'] = mdbo;    
        
    def forwardPass(self, inputSeq): #inputSeq is a sequence of input vectors (e.g. an input sentence)
        inputCount = len(inputSeq) #e.g. number of words in a sentence
        
        # The outputs at each time step.
        H = np.zeros((inputCount, self.hiddenDim))
        O = np.zeros((inputCount, self.hiddenDim))
        C = np.zeros((inputCount, self.hiddenDim))
        F = np.zeros((inputCount, self.hiddenDim))
        I = np.zeros((inputCount, self.hiddenDim))
        Z = np.zeros((inputCount, self.hiddenDim))
        
        softmaxPredictions = np.zeros((inputCount, self.hiddenDim))
        
        for t in range(inputCount): # for each item in a sequence (e.g. word in a sentence)
            xt = np.zeros(self.inputDim)
            xt[inputSeq[t]] = 1.0
            
            zt_ = np.dot(self.Wz, xt) + np.dot(self.Rz, H[t-1]) + self.bz
            zt = np.tanh(zt_)
            Z[t] = np.copy(zt)
            
            it_ = np.dot(self.Wi, xt) + np.dot(self.Ri, H[t-1]) + self.bi
            it = sigmoid(it_)
            I[t] = np.copy(it)
            
            ot_ = np.dot(self.Wo, xt) + np.dot(self.Ro, H[t-1]) + self.bo
            ot = sigmoid(ot_)
            O[t] = np.copy(ot)
            
            ft_ = np.dot(self.Wf, xt) + np.dot(self.Rf, H[t-1]) + self.bf
            ft = sigmoid(ft_)
            F[t] = np.copy(ft)
            
            
            ct = zt * it + C[t-1] * ft
            C[t] = np.copy(ct)
            
            ht = np.tanh(ct) * ot
            
            H[t] = np.copy(ht) 
            softmaxPredictions[t] = np.copy(self.stable_softmax(ht))
        
        return softmaxPredictions, H, O, C, F, I, Z
    
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
        '''
        L = 0
        softmaxPredictions, H, O, C, F, I, Z = self.forwardPass(inputSeq)
        for t in range(len(H)):
            # We only care about our prediction of the "correct" words
            #the first index is all the samples, and the second index is the correct word index
            #which give an array for all the correct output predictions
            correct_word_predictions = softmaxPredictions[np.arange(len(trueOutputSeq)), trueOutputSeq[t]]
            # Add to the loss based on how off we were
            #note that if we made correct predictions, then the output corresponding to the correct word is 1.0
            #which makes the loss = 0
            L += -1 * np.sum(np.log(correct_word_predictions)) # L = - sum{ 1.0 * log(pi)}
        return L
        '''
        L = 0
        softmaxPredictions, H, O, C, F, I, Z = self.forwardPass(inputSeq)
        for t in range(len(H)):
            L -= np.log(softmaxPredictions[t][trueOutputSeq[t]])
        
        return L

    
    def backProp(self, inputBatch, trueOutputBatch):
        deltas = {}
        dWz = np.zeros(self.Wz.shape)
        dWi = np.zeros(self.Wi.shape)
        dWf = np.zeros(self.Wf.shape)
        dWo = np.zeros(self.Wo.shape)
        
        dRz = np.zeros(self.Rz.shape)
        dRi = np.zeros(self.Ri.shape)
        dRf = np.zeros(self.Rf.shape)
        dRo = np.zeros(self.Ro.shape)
        
        dbz = np.zeros(self.bz.shape)
        dbi = np.zeros(self.bi.shape)
        dbf = np.zeros(self.bf.shape)
        dbo = np.zeros(self.bo.shape)
        
        for b in range(len(inputBatch)): #for each input sentence in the batch
            X = inputBatch[b]
            
            inputCount = len(X)
            softmaxPredictions, H, O, C, F, I, Z = self.forwardPass(X)
            
            T = len(X)
            
            dH = np.zeros((inputCount, self.hiddenDim))
            dO = np.zeros((inputCount, self.hiddenDim))
            dC = np.zeros((inputCount, self.hiddenDim))
            dF = np.zeros((inputCount, self.hiddenDim))
            dI = np.zeros((inputCount, self.hiddenDim))
            dZ = np.zeros((inputCount, self.hiddenDim))
            
            for t in reversed(range(T)): #from idx T-1 down to 0
                xt = np.zeros((self.inputDim))
                xt[X[t]] = 1.0
                Y = np.zeros(self.inputDim)
                Y[trueOutputBatch[b][t]] = 1.0
                dH[t] += softmaxPredictions[t] - Y
                if(t+1 < T):
                    dH[t] += np.dot(self.Rz, dZ[t+1]) + np.dot(self.Ri, dI[t+1]) + np.dot(self.Rf, dF[t+1]) + np.dot(self.Ro, dO[t+1])
                
                dO[t] += dH[t] * np.tanh(C[t]) * O[t] * (1 - O[t]) 
                
                dC[t] += dH[t] * O[t] * (1 - np.tanh(C[t]) * np.tanh(C[t]))
                if(t+1 < T):
                    dC[t] += dC[t+1] * F[t+1]
                
                if(t-1 >= 0):  
                    dF[t] += dC[t] * C[t-1] * F[t] * (1 - F[t])
                
                dI[t] += dC[t] * Z[t] * I[t] * (1 - I[t])
                
                dZ[t] += dC[t] * I[t] * (1 - Z[t]*Z[t])
                
                dWz += np.outer(dZ[t], xt)
                dWi += np.outer(dI[t], xt)
                dWf += np.outer(dF[t], xt)
                dWo += np.outer(dO[t], xt)
                
                
                if(t+1 < T):
                    dRz += np.outer(dZ[t+1], H[t])
                    dRi += np.outer(dI[t+1], H[t])
                    dRf += np.outer(dF[t+1], H[t])
                    dRo += np.outer(dO[t+1], H[t])
                
                
                
                '''
                if(t-1 >= 0):
                    dRz += np.outer(dZ[t], H[t-1])
                    dRi += np.outer(dI[t], H[t-1])
                    dRf += np.outer(dF[t], H[t-1])
                    dRo += np.outer(dO[t], H[t-1])
                '''
                dbz += dZ[t]
                dbi += dI[t]
                dbf += dF[t]
                dbo += dO[t]
                
            #print 'dgate ',np.linalg.norm(dZ), np.linalg.norm(dI), np.linalg.norm(dF), np.linalg.norm(dO)
        #print 'dweight ',np.linalg.norm(dWz), np.linalg.norm(dWi), np.linalg.norm(dWf), np.linalg.norm(dWo)
        
        #gradient clipping
  
        deltas['Wz'] = dWz; deltas['Wi'] = dWi; deltas['Wf'] = dWf; deltas['Wo'] = dWo
        deltas['Rz'] = dRz; deltas['Ri'] = dRi; deltas['Rf'] = dRf; deltas['Ro'] = dRo
        deltas['bz'] = dbz; deltas['bi'] = dbi; deltas['bf'] = dbf; deltas['bo'] = dbo
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
            self.Wz = self.Wz - self.learningRate * deltas['Wz'] / np.sqrt(self.memory['dWz'] + 1e-8)
            self.Wi = self.Wi - self.learningRate * deltas['Wi'] / np.sqrt(self.memory['dWi'] + 1e-8)
            self.Wf = self.Wf - self.learningRate * deltas['Wf'] / np.sqrt(self.memory['dWf'] + 1e-8)
            self.Wo = self.Wo - self.learningRate * deltas['Wo'] / np.sqrt(self.memory['dWo'] + 1e-8)
            
            self.Rz = self.Rz - self.learningRate * deltas['Rz'] / np.sqrt(self.memory['dRz'] + 1e-8)
            self.Ri = self.Ri - self.learningRate * deltas['Ri'] / np.sqrt(self.memory['dRi'] + 1e-8)
            self.Rf = self.Rf - self.learningRate * deltas['Rf'] / np.sqrt(self.memory['dRf'] + 1e-8)
            self.Ro = self.Ro - self.learningRate * deltas['Ro'] / np.sqrt(self.memory['dRo'] + 1e-8)
            
            self.bz = self.bz - self.learningRate * deltas['bz'] / np.sqrt(self.memory['dbz'] + 1e-8)
            self.bi = self.bi - self.learningRate * deltas['bi'] / np.sqrt(self.memory['dbi'] + 1e-8)
            self.bf = self.bf - self.learningRate * deltas['bf'] / np.sqrt(self.memory['dbf'] + 1e-8)
            self.bo = self.bo - self.learningRate * deltas['bo'] / np.sqrt(self.memory['dbo'] + 1e-8)
        else:
            self.Wz = self.Wz - self.learningRate * deltas['Wz'] 
            self.Wi = self.Wi - self.learningRate * deltas['Wi'] 
            self.Wf = self.Wf - self.learningRate * deltas['Wf'] 
            self.Wo = self.Wo - self.learningRate * deltas['Wo']
            
            self.Rz = self.Rz - self.learningRate * deltas['Rz'] 
            self.Ri = self.Ri - self.learningRate * deltas['Ri']
            self.Rf = self.Rf - self.learningRate * deltas['Rf']
            self.Ro = self.Ro - self.learningRate * deltas['Ro']
            
            self.bz = self.bz - self.learningRate * deltas['bz']
            self.bi = self.bi - self.learningRate * deltas['bi']
            self.bf = self.bf - self.learningRate * deltas['bf']
            self.bo = self.bo - self.learningRate * deltas['bo']
    

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
    ct_1 = np.zeros(lstm.hiddenDim)
    while wordCount > 0:
        xt = np.zeros(lstm.inputDim)
        xt[w2i[genWord]] = 1.0
        softmaxPredictions, ht, ct = lstm.generate(xt, ht_1, ct_1)
        #print sum(softmaxPredictions[0])
        #genWordId = np.random.choice(range(lstm.inputDim), p=softmaxPredictions[0].ravel())
        genWordId = softmaxPredictions[0].argmax()
        genWord = i2w[genWordId]
        sent += genWord + ','
        ht_1 = ht
        ct_1 = ct
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
    
    
    lstm = zLSTM(inputDim=D, hiddenDim=H, learningRate=0.01, clipGradients= False, useAdaGrad=False, batchSize = 1)
    
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
            
            print 'Generated chars:\n', generateText(lstm, w2i, i2w, ' ', 50)
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