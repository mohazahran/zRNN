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
        self.bf = np.ones((1,hiddenDim))
        #self.bf = np.random.uniform(-0.1, 0.1, (1,hiddenDim)) #better to be initialized to all ones.
        self.bo = np.random.uniform(-0.1, 0.1, (1,hiddenDim))
        
    def forwardPass(self, inputSeq): #inputSeq is a sequence of input vectors (e.g. an input sentence)
        inputCount = len(inputSeq) #e.g. number of words in a sentence
        
        # The outputs at each time step. Again, we save them for later.
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
            zt = sigmoid(zt_)
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
            softmaxPredictions[t] = self.stable_softmax(ht)
        
        return softmaxPredictions, H, O, C, F, I, Z
    
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
            Y = np.zeros(self.inputDim)
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
                Y[trueOutputBatch[b][t]] = 1.0
                dH[t] += softmaxPredictions[t] - Y
                if(t+1 < T):
                    dH[t] += np.dot(self.Rz, dZ[t+1]) + np.dot(self.Ri, dI[t+1]) + np.dot(self.Rf, dF[t+1]) + np.dot(self.Ro, dO[t+1])
                
                dO[t] += dH[t] * np.tanh(C[t]) * O[t] * (1. - O[t])
                
                dC[t] += dH[t] * O[t] * (1. - np.tanh(C[t])**2)
                if(t+1 < T):
                    dC[t] += dC[t+1] * F[t+1]
                    
                dF[t] += dC[t] * C[t-1] * F[t] * (1. - F[t])
                
                dI[t] += dC[t] * Z[t] * I[t] * (1. - I[t])
                
                dZ[t] += dC[t] * I[t] * Z[t] * (1. - Z[t])
                
                dWz += np.outer(dZ[t], X[t])
                dWi += np.outer(dI[t], X[t])
                dWf += np.outer(dF[t], X[t])
                dWo += np.outer(dO[t], X[t])
                
                if(t+1 < T):
                    dRz += np.outer(dZ[t+1], H[t])
                    dRi += np.outer(dI[t+1], H[t])
                    dRf += np.outer(dF[t+1], H[t])
                    dRo += np.outer(dO[t+1], H[t])
                
                dbz += dZ[t]
                dbi += dI[t]
                dbf += dF[t]
                dbo += dO[t]
                
            #print 'dgate ',np.linalg.norm(dZ), np.linalg.norm(dI), np.linalg.norm(dF), np.linalg.norm(dO)
        #print 'dweight ',np.linalg.norm(dWz), np.linalg.norm(dWi), np.linalg.norm(dWf), np.linalg.norm(dWo)
        
        #gradient clipping
        '''
        gradientLimit = 5.
        dWz[dWz > gradientLimit] = gradientLimit
        dWz[dWz < -gradientLimit] = -gradientLimit
        dWi[dWi > gradientLimit] = gradientLimit
        dWi[dWi < -gradientLimit] = -gradientLimit
        dWf[dWf > gradientLimit] = gradientLimit
        dWf[dWf < -gradientLimit] = -gradientLimit
        dWo[dWo > gradientLimit] = gradientLimit
        dWo[dWo < -gradientLimit] = -gradientLimit
        
        dRz[dRz > gradientLimit] = gradientLimit
        dRz[dRz < -gradientLimit] = -gradientLimit
        dRi[dRi > gradientLimit] = gradientLimit
        dRi[dRi < -gradientLimit] = -gradientLimit
        dRf[dRf > gradientLimit] = gradientLimit
        dRf[dRf < -gradientLimit] = -gradientLimit
        dRo[dRo > gradientLimit] = gradientLimit
        dRo[dRo < -gradientLimit] = -gradientLimit
        
        dbz[dbz > gradientLimit] = gradientLimit
        dbz[dbz < -gradientLimit] = -gradientLimit
        dbi[dbi > gradientLimit] = gradientLimit
        dbi[dbi < -gradientLimit] = -gradientLimit
        dbf[dbf > gradientLimit] = gradientLimit
        dbf[dbf < -gradientLimit] = -gradientLimit
        dbo[dbo > gradientLimit] = gradientLimit
        dbo[dbo < -gradientLimit] = -gradientLimit
        '''
                
        self.Wz = self.Wz - self.learningRate * dWz/float(len(inputBatch))
        self.Wi = self.Wi - self.learningRate * dWi/float(len(inputBatch))
        self.Wf = self.Wf - self.learningRate * dWf/float(len(inputBatch))
        self.Wo = self.Wo - self.learningRate * dWo/float(len(inputBatch))
        
        self.Rz = self.Rz - self.learningRate * dRz/float(len(inputBatch))
        self.Ri = self.Ri - self.learningRate * dRi/float(len(inputBatch))
        self.Rf = self.Rf - self.learningRate * dRf/float(len(inputBatch))
        self.Ro = self.Ro - self.learningRate * dRo/float(len(inputBatch))
        
        self.bz = self.bz - self.learningRate * dbz/float(len(inputBatch))
        self.bi = self.bi - self.learningRate * dbi/float(len(inputBatch))
        self.bf = self.bf - self.learningRate * dbf/float(len(inputBatch))
        self.bo = self.bo - self.learningRate * dbo/float(len(inputBatch))
        
    

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
    D = 50 # Number of input dimension == number of items in vocabulary
    H = D # Number of LSTM layer's neurons
    epochs = 50
    valQuota = 0.2
    
    X_all, Y_all = preProcessing(D)
    valSize = int(valQuota * len(X_all))
    X_val = X_all[:valSize]
    Y_val = Y_all[:valSize]
    X_train = X_all[valSize:]
    Y_train = Y_all[valSize:]
    lstm = zLSTM(D, H)
    lr = 0.5
    for i in range(epochs):
        print 'Epoch#',i
        
        crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
        print 'Cross Entropy TRAIN Loss before= ', crossEntropyLoss
        
        crossEntropyLoss = lstm.calculate_loss_batch(X_val, Y_val)
        print 'Cross Entropy VAL Loss before= ', crossEntropyLoss
        
        #lstm.train(X_train, Y_train, batchSize=10, learningRate=lr/float(i+1))
        lstm.train(X_train, Y_train, batchSize=10, learningRate=lr)
        
        crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
        print 'Cross Entropy TRAIN Loss after=  ', crossEntropyLoss
        
        crossEntropyLoss = lstm.calculate_loss_batch(X_val, Y_val)
        print 'Cross Entropy VAL Loss after=  ', crossEntropyLoss
        
        '''
        lstm.Wz = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        lstm.Wi = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        lstm.Wf = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        lstm.Wo = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        
        lstm.Rz = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        lstm.Ri = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        lstm.Rf = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        lstm.Ro = np.random.uniform(-0.1, 0.1, (lstm.hiddenDim, lstm.inputDim))
        
        lstm.bz = np.random.uniform(-0.1, 0.1, (1,lstm.hiddenDim))
        lstm.bi = np.random.uniform(-0.1, 0.1, (1,lstm.hiddenDim))
        lstm.bf = np.ones((1,lstm.hiddenDim))
        #self.bf = np.random.uniform(-0.1, 0.1, (1,hiddenDim)) #better to be initialized to all ones.
        lstm.bo = np.random.uniform(-0.1, 0.1, (1,lstm.hiddenDim))
        '''

if __name__ == '__main__':
    main()