'''
Created on Feb 21, 2018

@author: mohame11
'''
import numpy as np
import nltk
import csv
import itertools
import math
from random import shuffle

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class zLSTM(object):
    '''
    classdocs
    '''
    

    def __init__(self, inputDim, hiddenDim):
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        
        self.Wz = np.array([0.45, 0.25])
        self.Wi = np.array([0.95, 0.8])
        self.Wf = np.array([0.7, 0.45])
        self.Wo = np.array([0.6, 0.4])
        
        self.Rz = np.array([0.15])
        self.Ri = np.array([0.8])
        self.Rf = np.array([0.1])
        self.Ro = np.array([0.25])
        
        self.bz = np.array([0.2])
        self.bi = np.array([0.65])
        self.bf = np.array([0.15])
        self.bo = np.array([0.1])
        
    def forwardPass(self, inputSeq): #inputSeq is a sequence of input vectors (e.g. an input sentence)
        inputCount = len(inputSeq)
        
        # The outputs at each time step. Again, we save them for later.
        H = np.zeros((inputCount, self.hiddenDim))
        O = np.zeros((inputCount, self.hiddenDim))
        C = np.zeros((inputCount, self.hiddenDim))
        F = np.zeros((inputCount, self.hiddenDim))
        I = np.zeros((inputCount, self.hiddenDim))
        Z = np.zeros((inputCount, self.hiddenDim))
        predictions = np.zeros((inputCount, self.hiddenDim))
        for t in range(inputCount):
            xt = inputSeq[t]
            
            zt_ = np.dot(self.Wz, xt) + np.dot(self.Rz, H[t-1]) + self.bz
            zt = np.tanh(zt_)
            Z[t] = zt
            
            it_ = np.dot(self.Wi, xt) + np.dot(self.Ri, H[t-1]) + self.bi
            it = sigmoid(it_)
            I[t] = it
            
            ot_ = np.dot(self.Wo, xt) + np.dot(self.Ro, H[t-1]) + self.bo
            ot = sigmoid(ot_)
            O[t] = ot
            
            ft_ = np.dot(self.Wf, xt) + np.dot(self.Rf, H[t-1]) + self.bf
            ft = sigmoid(ft_)
            F[t] = ft
            
            
            ct = zt * it + C[t-1] * ft
            C[t] = ct
            
            ht = np.tanh(ct) * ot
            
            H[t] = ht # save the current output at time t
            predictions[t] = ht
        
        return predictions, H, O, C, F, I, Z
    
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
            L -= np.log(softmaxPredictions[t][trueOutputSeq[t]])
        
        return L
        '''
        L = 0
        softmaxPredictions, H, O, C, F, I, Z = self.forwardPass(inputSeq)
        for t in range(len(H)):
            L += 0.5*(softmaxPredictions[t]-[trueOutputSeq[t]])**2
        
        return L
        
        
    
    
    def backProp(self, inputBatch, trueOutputBatch):
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
        
        X = inputBatch
        inputCount = len(X)
        softmaxPredictions, H, O, C, F, I, Z = self.forwardPass(X)
        T = len(H)
        
        dH = np.zeros((inputCount, self.hiddenDim))
        dO = np.zeros((inputCount, self.hiddenDim))
        dC = np.zeros((inputCount, self.hiddenDim))
        dF = np.zeros((inputCount, self.hiddenDim))
        dI = np.zeros((inputCount, self.hiddenDim))
        dZ = np.zeros((inputCount, self.hiddenDim))
        
        for t in range(T-1,-1,-1): #from T down to 0
            Y = trueOutputBatch[t]
            dH[t] += softmaxPredictions[t] - Y 
            if(t+1 < T):
                dH[t] += np.dot(self.Rz, dZ[t+1]) + np.dot(self.Ri, dI[t+1]) + np.dot(self.Rf, dF[t+1]) + np.dot(self.Ro, dO[t+1])
            
            dO[t] += dH[t] * np.tanh(C[t]) * O[t] * (1. - O[t])
            
            dC[t] += dH[t] * O[t] * (1. - np.tanh(C[t])**2)
            if(t+1 < T):
                dC[t] += dC[t+1] * F[t+1]
            
            if(t-1 >= 0):
                dF[t] += dC[t] * C[t-1] * F[t] * (1. - F[t])
            
            dI[t] += dC[t] * Z[t] * I[t] * (1. - I[t])
            
            dZ[t] += dC[t] * I[t] * (1. - (Z[t])**2)
            
            dWz += np.outer(dZ[t], X[t])[0]
            dWi += np.outer(dI[t], X[t])[0]
            dWf += np.outer(dF[t], X[t])[0]
            dWo += np.outer(dO[t], X[t])[0]
            
            if(t+1 < T):
                dRz += np.outer(dZ[t+1], H[t])[0]
                dRi += np.outer(dI[t+1], H[t])[0]
                dRf += np.outer(dF[t+1], H[t])[0]
                dRo += np.outer(dO[t+1], H[t])[0]
            
            dbz += dZ[t]
            dbi += dI[t]
            dbf += dF[t]
            dbo += dO[t]
                
        
                
        self.Wz = self.Wz - self.learningRate * dWz
        self.Wi = self.Wi - self.learningRate * dWi
        self.Wf = self.Wf - self.learningRate * dWf
        self.Wo = self.Wo - self.learningRate * dWo
        
        self.Rz = self.Rz - self.learningRate * dRz
        self.Ri = self.Ri - self.learningRate * dRi
        self.Rf = self.Rf - self.learningRate * dRf
        self.Ro = self.Ro - self.learningRate * dRo
        
        self.bz = self.bz - self.learningRate * dbz
        self.bi = self.bi - self.learningRate * dbi
        self.bf = self.bf - self.learningRate * dbf
        self.bo = self.bo - self.learningRate * dbo
        
    

    def stable_softmax(self, X):
        #table softmax
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    
        #exps = np.exp(X)
        #return exps / np.sum(exps)

    def train(self, trainingSet = [], trainingTruth = [], batchSize = 10, learningRate = 0.5):
        self.learningRate = learningRate
        batchCount = int(math.ceil(float(len(trainingSet)) /batchSize))
        for b in range(batchCount):
            X_batch = trainingSet[b*batchSize : (b+1)*batchSize]
            Y_batch = trainingTruth[b*batchSize : (b+1)*batchSize]
            self.backProp(X_batch, Y_batch)
            
    
    

def preProcessing(vocabSize):
    vocabulary_size = vocabSize
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    with open('redditText', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
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
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
     
    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
     
    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, Y_train

def main():
    np.random.seed(100)
    
    x0 = np.array([1., 2.]); y0 = np.array([0.5])
    x1 = np.array([0.5 ,3.]); y1 = np.array([1.25])
    
    X_train = np.array([x0, x1])
    Y_train = np.array([y0, y1])
    
    lstm = zLSTM(2, 1)
    lr = 0.1
    for i in range(5):
        print 'Epoch#',i
        
        crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
        print 'Cross Entropy TRAIN Loss = ', crossEntropyLoss
        
        lstm.train(X_train, Y_train, batchSize=2, learningRate=lr)
        
        

if __name__ == '__main__':
    main()