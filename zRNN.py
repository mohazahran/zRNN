'''
Created on Mar 4, 2018

@author: mohame11
'''
import numpy as np
import operator


class zRNN(object):
    '''
    classdocs
    '''


    def __init__(self, inputDim, hiddenDim=100, bptt_truncate=4):
        # Assign instance variables
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./inputDim), np.sqrt(1./inputDim), (hiddenDim, inputDim))
        self.V = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (inputDim, hiddenDim))
        self.W = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (hiddenDim, hiddenDim))
        
    def forward_propagation(self, x):
        # The total number of time steps
        wordCount = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((wordCount + 1, self.hiddenDim)) #wordCount+1 to have an initial state for the first word
        s[-1] = np.zeros(self.hiddenDim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((wordCount, self.inputDim))
        # For each time step...
        for t in np.arange(wordCount):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = self.stable_softmax(self.V.dot(s[t]))
        return [o, s]
    
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    
    def calculate_total_loss(self, X, Y):
        L = 0
        # For each sentence...
        for i in np.arange(len(Y)):
            o, s = self.forward_propagation(X[i])
            # We only care about our prediction of the "correct" words
            #the first index is all the samples, and the second index is the correct word index
            #which give an array for all the correct output predictions
            correct_word_predictions = o[np.arange(len(Y[i])), Y[i]]
            # Add to the loss based on how off we were
            #note that if we made correct predictions, then the output corresponding to the correct word is 1.0
            #which makes the loss = 0
            L += -1 * np.sum(np.log(correct_word_predictions)) # L = - sum{ 1.0 * log(pi)}
        return L
 
    def calculate_loss(self, X, Y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in Y))
        return self.calculate_total_loss(X,Y)/N
    
    def bptt(self, x, y):
        wordCount = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1. #dL/do = delta_o = oi - yi, yi = 1
        # For each output backwards...
        for t in np.arange(wordCount)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
 
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error and error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)