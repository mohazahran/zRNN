'''
Created on Mar 3, 2018

@author: mohame11
'''
from zUnit import *
from zLSTM import *
from zRNN import *
import numpy as np
import nltk
import csv
import itertools

vocabulary_size = 500
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
    
def preProcessing():
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    with open('redditText_sample', 'rb') as f:
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
    np.random.seed(10)
    X_train, Y_train = preProcessing()
    
    #model = zRNN(inputDim = vocabulary_size, hiddenDim = 100, bptt_truncate=4)
    #model.calculate_loss(X_train, Y_train)
    
    model = zRNN(100, 10, bptt_truncate=10)
    model.gradient_check(X_train, Y_train)
    

    






if __name__ == '__main__':
    main()