import re 
import string 
import sys
import numpy as np
import math
import time

import json
import sys
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import lru_cache


__author__= 'KD'


def json_writer(data, fname):
    """
        Write multiple json files
        Args:
            data: list(dict): list of dictionaries to be written as json
            fname: str: output file name
    """
    with open(fname, mode="w") as fp:
        for line in data:
            json.dump(line, fp)
            fp.write("\n")


def json_reader(fname):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    for line in open(fname, mode="r"):
        yield json.loads(line)


def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)


def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            #print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    ps = PorterStemmer()
    #ps_stem = lru_cache(maxsize=None)(ps.stem) # use this function to stem
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, ps, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, ps, en_stop, return_tokens)






train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

labels = 5 # the number of labels we are classifying on
label_classes = [1,2,3,4,5]

train_generator = json_reader(train_file)
test_generator = json_reader(test_file)

word_stemming_map = {}

word_index_dict_stem_bigrams = {}

len_words_stemmed_bigrams = 0

# This functions converts our data into our desired form(a list of integers for each document where the integer at ith index is the index of the word in the vocabulary)
def retrieve_data(generator):
    global len_words_stemmed_bigrams
    
    train_y = []

    stemmed_bigrams = []

    count = 0

    for line in generator:
        stemmed_bigram = []
        my_feature = []

        res = re.sub('['+string.punctuation+']', '', line['text']).split()
        rating = int(line['stars'])
        train_y.append(rating)

        prev_word = ''
        prev_word_my_feature = ''

        for word in res:

            if word not in word_stemming_map:
                word_stemming_map[word] = getStemmedDocuments(word)

            stemmed_word = word_stemming_map[word]

            if not stemmed_word:
                continue

            n = len(stemmed_word)

            for i in range(n):
                w = stemmed_word[i]

                bigram = prev_word + ' ' + w

                if bigram not in word_index_dict_stem_bigrams:
                    word_index_dict_stem_bigrams[bigram] = len_words_stemmed_bigrams
                    stemmed_bigram.append(len_words_stemmed_bigrams)
                    len_words_stemmed_bigrams += 1
                else:
                    stemmed_bigram.append(word_index_dict_stem_bigrams[bigram])

                prev_word = w

        stemmed_bigrams.append(stemmed_bigram)

        #count += 1
        #if(count == 10000):
            #break

    return (stemmed_bigrams,train_y)

start_t1 = time.time()

start_t = time.time()
(train_stemmed_bigrams,train_y) = retrieve_data(train_generator)
end_t1 = time.time()
#print('Train Data reading complete')
#print('Stemming of training data complete')
#print('Time taken: ' , end_t1 - start_t , ' seconds')

start_t = time.time()
(test_stemmed_bigrams,test_y) = retrieve_data(test_generator)
end_t1 = time.time()
#print('Test Data reading complete')
#print('Stemming of test data complete')
#print('Time taken: ' , end_t1 - start_t , ' seconds')

#print('-----------------------------------------------------------------')

word_stemming_map.clear()
word_index_dict_stem_bigrams.clear()


class NaiveBayes:
    def train_model(self,train_x,train_y,len_words):
        unique, counts = np.unique(train_y, return_counts=True)
        self.logphik = np.zeros(labels)
        m = len(train_y)
        logm = math.log(m)
        for i in range(labels):
            self.logphik[i] = (math.log(counts[i]) - logm)
        
        # theta(l/k) = theta[k][l]

        denom = np.full(labels , len_words)
        numer = np.ones((labels,len_words))

        for i in range(m):
            n_i = len(train_x[i])
            denom[train_y[i] - 1] += n_i
            for j in range(n_i):
                numer[train_y[i] - 1][train_x[i][j]] += 1

        self.logtheta = np.log(numer/denom[:,None])

        return self

    def predict_model(self,x):
        max_prob = float("-inf")
        max_label = 0

        for k in range(labels):
            prob = self.logphik[k]
            prob += np.sum(self.logtheta[k][x])

            if(prob > max_prob):
                max_prob = prob
                max_label = k+1

        return max_label

    def calc_accuracy(self,test_x,test_y):
        n = len(test_y)
        correct = 0
        
        for i in range(n):
            pred = self.predict_model(test_x[i])

            if(pred == test_y[i]):
                correct += 1

        return correct/n

    def make_predictions(self,test_x):
        n = len(test_x)
        predictions = []

        for i in range(n):
            pred = self.predict_model(test_x[i])

            predictions.append(int(pred))

        return predictions


bigram_model = NaiveBayes()
start_t = time.time()
bigram_model.train_model(train_stemmed_bigrams,train_y,len_words_stemmed_bigrams)
end_t1 = time.time()
#print('Time to train the stemmed bigram model:' , end_t1 - start_t , 'seconds')

#print('Test Accuracy(Stemmed Bigram):' , bigram_model.calc_accuracy(test_stemmed_bigrams,test_y)*100 , '%')

#print('-----------------------------------------------------------------')

#end_t2 = time.time()
##print('Total Time:' , end_t2 - start_t1 , 'seconds')

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

predictions = bigram_model.make_predictions(test_stemmed_bigrams)
write_predictions(output_file,predictions)






