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

#word_index_dict = {}
#word_index_dict_stem = {}
word_index_dict_stem_bigrams = {}
#word_index_dict_my_features = {}

#len_words = 0
#len_words_stemmed = 0
len_words_stemmed_bigrams = 0
#len_words_my_features = 0

# This functions converts our data into our desired form(a list of integers for each document where the integer at ith index is the index of the word in the vocabulary)
def retrieve_data(generator):
    #global len_words
    #global len_words_stemmed
    global len_words_stemmed_bigrams
    #global len_words_my_features

    #train_x = []
    #stemmed_x = []
    train_y = []

    stemmed_bigrams = []
    #my_features = []

    count = 0

    for line in generator:
        #x = []
        #stemmed_res = []
        stemmed_bigram = []
        my_feature = []

        res = re.sub('['+string.punctuation+']', '', line['text']).split()
        #res = re.sub(r'[^\w\s]', '', line['text']).split()
        rating = int(line['stars'])
        train_y.append(rating)

        prev_word = ''
        prev_word_my_feature = ''

        for word in res:
            """
            if word not in word_index_dict:
                word_index_dict[word] = len_words
                x.append(len_words)
                len_words += 1
            else:
                x.append(word_index_dict[word])
            """

            if word not in word_stemming_map:
                word_stemming_map[word] = getStemmedDocuments(word)

            stemmed_word = word_stemming_map[word]

            if not stemmed_word:
                continue

            n = len(stemmed_word)

            for i in range(n):
                w = stemmed_word[i]
                """
                if w not in word_index_dict_stem:
                    word_index_dict_stem[w] = len_words_stemmed
                    stemmed_res.append(len_words_stemmed)
                    len_words_stemmed += 1
                else:
                    stemmed_res.append(word_index_dict_stem[w])
                """

                bigram = prev_word + ' ' + w

                if bigram not in word_index_dict_stem_bigrams:
                    word_index_dict_stem_bigrams[bigram] = len_words_stemmed_bigrams
                    stemmed_bigram.append(len_words_stemmed_bigrams)
                    len_words_stemmed_bigrams += 1
                else:
                    stemmed_bigram.append(word_index_dict_stem_bigrams[bigram])

                prev_word = w

                """
                if(len(w) <= 3):
                    continue

                my_feature_word = prev_word_my_feature + ' ' + w

                if my_feature_word not in word_index_dict_my_features:
                    word_index_dict_my_features[my_feature_word] = len_words_my_features
                    my_feature.append(len_words_my_features)
                    len_words_my_features += 1
                else:
                    my_feature.append(word_index_dict_my_features[my_feature_word])

                prev_word_my_feature = w
                """


        #train_x.append(x)
        #stemmed_x.append(stemmed_res)
        stemmed_bigrams.append(stemmed_bigram)
        #my_features.append(my_feature)

        #count += 1
        #if(count == 10000):
            #break

    #return (train_x,train_y,stemmed_x,stemmed_bigrams,my_features)
    return (stemmed_bigrams,train_y)

start_t1 = time.time()

start_t = time.time()
#(train_x,train_y,train_stemmed_x,train_stemmed_bigrams,train_my_features) = retrieve_data(train_generator)
(train_stemmed_bigrams,train_y) = retrieve_data(train_generator)
end_t1 = time.time()
#print('Train Data reading complete')
#print('Stemming of training data complete')
#print('Time taken: ' , end_t1 - start_t , ' seconds')

start_t = time.time()
#(test_x,test_y,test_stemmed_x,test_stemmed_bigrams,test_my_features) = retrieve_data(test_generator)
(test_stemmed_bigrams,test_y) = retrieve_data(test_generator)
end_t1 = time.time()
#print('Test Data reading complete')
#print('Stemming of test data complete')
#print('Time taken: ' , end_t1 - start_t , ' seconds')

#print('-----------------------------------------------------------------')

word_stemming_map.clear()
#word_index_dict.clear()
#word_index_dict_stem.clear()
word_index_dict_stem_bigrams.clear()
#word_index_dict_my_features.clear()

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
            #n_i = train_x[i].shape[0]
            denom[train_y[i] - 1] += n_i
            for j in range(n_i):
                numer[train_y[i] - 1][train_x[i][j]] += 1

        self.logtheta = np.log(numer/denom[:,None])
        #self.unassigned_k = -np.log(denom)

        return self

    def predict_model(self,x,i):
        max_prob = float("-inf")
        max_label = 0

        for k in range(labels):
            prob = self.logphik[k]
            prob += np.sum(self.logtheta[k][x])

            #self.score[i][k] = prob

            if(prob > max_prob):
                max_prob = prob
                max_label = k+1

        """
        sum_prob = 0
        for k in range(labels):
            self.score[i][k] = math.exp(self.score[i][k] - max_prob)
            sum_prob += self.score[i][k]
        """

        #self.score[i]/=sum_prob

        return max_label

    def calc_accuracy(self,test_x,test_y,confusion=False):
        n = len(test_y)
        correct = 0

        #if confusion:
            #self.confusion_matrix = np.zeros((labels,labels) , dtype = int)

        #self.score = np.zeros((n,labels))
        
        for i in range(n):
            pred = self.predict_model(test_x[i],i)
            #if confusion:
                #self.confusion_matrix[pred - 1][test_y[i] - 1] += 1
            if(pred == test_y[i]):
                correct += 1

        return correct/n

# decomment the following code to train the raw model based on words
"""
raw_model = NaiveBayes()
start_t = time.time()
raw_model.train_model(train_x,train_y,len_words)
end_t1 = time.time()
#print('Time to train the raw model:' , end_t1 - start_t , 'seconds')

#print('Train Accuracy(Raw):' , raw_model.calc_accuracy(train_x,train_y)*100 , '%')
start_t = time.time()
#print('Test Accuracy(Raw):' , raw_model.calc_accuracy(test_x,test_y,True)*100 , '%')
end_t1 = time.time()
#print('Test Accuracy(Random): ' , 100/labels , '%')
#print('Test Accuracy(Max Occurance): ' , (np.amax(np.unique(test_y, return_counts=True)[1])/len(test_y))*100 , '%')
#print('Confusion Matrix(Raw):')
#print(raw_model.confusion_matrix)

#print('-----------------------------------------------------------------')
"""


# decomment the following to train the model based on stemmed words
"""
stemmed_model = NaiveBayes()
start_t = time.time()
stemmed_model.train_model(train_stemmed_x,train_y,len_words_stemmed)
end_t1 = time.time()
#print('Time to train the stemmed model:' , end_t1 - start_t , 'seconds')

#print('Test Accuracy(Stemmed):' , stemmed_model.calc_accuracy(test_stemmed_x,test_y)*100 , '%')

#print('-----------------------------------------------------------------')
"""

bigram_model = NaiveBayes()
start_t = time.time()
bigram_model.train_model(train_stemmed_bigrams,train_y,len_words_stemmed_bigrams)
end_t1 = time.time()
#print('Time to train the stemmed bigram model:' , end_t1 - start_t , 'seconds')

#print('Test Accuracy(Stemmed Bigram):' , bigram_model.calc_accuracy(test_stemmed_bigrams,test_y)*100 , '%')

#print('-----------------------------------------------------------------')

#end_t2 = time.time()
##print('Total Time:' , end_t2 - start_t1 , 'seconds')

"""
my_features_model = NaiveBayes()
start_t = time.time()
my_features_model.train_model(train_my_features,train_y,len_words_my_features)
end_t1 = time.time()
#print('Time to train the my features model:' , end_t1 - start_t , 'seconds')

#print('Test Accuracy(My features):' , my_features_model.calc_accuracy(test_my_features,test_y)*100 , '%')

#print('-----------------------------------------------------------------')
"""


"""
# decomment the following code to plot roc curve(assuming the stemmed bigram model has already been trained)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

def plot_roc(train_y,test_y,bigram_model):
    train_y_arr = np.array(train_y)
    test_y_arr = np.array(test_y)

    m = train_y_arr.shape[0]
    total_y = np.concatenate((train_y_arr,test_y_arr))
    y = label_binarize(total_y, classes=label_classes)
    n_classes = y.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[m:][:, i], bigram_model.score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y[m:].ravel(), bigram_model.score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

plot_roc(train_y,test_y,bigram_model)
"""

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")







