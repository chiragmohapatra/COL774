import re 
import string 
import sys
import numpy as np
import math
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import SGDClassifier

import json

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

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_generator = json_reader(train_file)
test_generator = json_reader(test_file)

def retrieve_train_data(generator):
    train_x = []
    train_y = []

    val_x = []
    val_y = []

    count = 0

    for line in generator:
        res = re.sub('['+string.punctuation+']', '', line['text'])
        rating = int(line['stars'])

        count += 1

        if count % 5 == 0:
            val_y.append(rating)
            val_x.append(res)
        else:
            train_y.append(rating)
            train_x.append(res)

        #if(count == 10000):
            #break

    return (train_x,np.array(train_y),val_x,np.array(val_y))

def retrieve_test_data(generator):
    train_x = []
    train_y = []

    count = 0

    for line in generator:
        res = re.sub('['+string.punctuation+']', '', line['text'])
        rating = int(line['stars'])

        train_y.append(rating)
        train_x.append(res)

        """
        count += 1
        if(count == 10000):
            break
        """

    return (train_x,np.array(train_y))


(train_data,train_y,val_data,val_y) = retrieve_train_data(train_generator)
print('Train reading complete')

(test_data,test_y) = retrieve_test_data(test_generator)

def transform_data(train_data,test_data,val_data):
    count_vect = CountVectorizer()
    train_cnt = count_vect.fit_transform(train_data)
    test_cnt = count_vect.transform(test_data)
    val_cnt = count_vect.transform(val_data)

    tfidf_transformer = TfidfTransformer()
    train_x = tfidf_transformer.fit_transform(train_cnt)

    test_x = tfidf_transformer.transform(test_cnt)

    val_x = tfidf_transformer.transform(val_cnt)

    return (train_x,test_x,val_x)

def calc_accuracy_NB_model(NB_model,test_x,test_y):
    pred = NB_model.predict(test_x)
    return np.mean(test_y == pred)

start = time.time()
(train_x,test_x,val_x) = transform_data(train_data,test_data,val_data)
end = time.time()
print('Transformation of data complete in ' , (end - start) , ' seconds')

"""
start = time.time()
NB_model = MultinomialNB().fit(train_cnt, train_y)
end = time.time()
print('Training of Naive Bayes complete in ' , (end - start) , ' seconds')
NB_accuracy = calc_accuracy_NB_model(NB_model,test_cnt,test_y)

print('Test Accuracy of Naive Bayes model: ' , NB_accuracy*100 , '%')
"""

def hyper_tuning_c(C_list):
    max_c = 0
    max_accur = 0

    for c in C_list:
        start = time.time()
        SVM_model = make_pipeline(LinearSVC(random_state=0, tol=1e-5 , C=c))
        SVM_model.fit(train_x,train_y)
        end = time.time()
        #print('C = ' , c)
        #print('Training of SVM complete in ' , (end - start) , ' seconds')

        SVM_accuracy = SVM_model.score(val_x,val_y)

        #print('Test Accuracy of SVM model: ' , SVM_accuracy*100 , '%')

        if SVM_accuracy > max_accur:
            max_accur = SVM_accuracy
            max_c = c

    return max_c

C = hyper_tuning_c([0.1,0.5,1])
print(C)
SVM_model = make_pipeline(LinearSVC(random_state=0, tol=1e-5 , C=C))
SVM_model.fit(train_x,train_y)

test_pred = SVM_model.predict(test_x)

print(np.mean(test_pred == test_y))

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

write_predictions(output_file,test_pred)


"""
for t in [1e-2,1e-3,1e-4,1e-5]:
    start = time.time()
    SGD_model = make_pipeline(SGDClassifier(max_iter=1000, tol=t))
    #SGD_model = make_pipeline(SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42))
    SGD_model.fit(train_x,train_y)
    end = time.time()
    print('tol = ' , t)
    print('Training of SGD Classifier complete in ' , (end - start) , ' seconds')

    SGD_accuracy = SGD_model.score(test_x,test_y)

    print('Test Accuracy of SGD model: ' , SGD_accuracy*100 , '%')

for a in [1e-2,1e-3,1e-4,1e-5]:
    start = time.time()
    SGD_model = make_pipeline(SGDClassifier(max_iter=1000, tol=1e-4 , alpha=a))
    #SGD_model = make_pipeline(SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42))
    SGD_model.fit(train_x,train_y)
    end = time.time()
    print('tol = ' , t)
    print('Training of SGD Classifier complete in ' , (end - start) , ' seconds')

    SGD_accuracy = SGD_model.score(test_x,test_y)

    print('Test Accuracy of SGD model: ' , SGD_accuracy*100 , '%')
"""

