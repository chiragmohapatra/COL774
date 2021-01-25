import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import sys

import time

k = 10 # the number of labels 
gamma = 0.05 # parameter for Gaussian Kernel
epsilon = 1e-05 # the parameter to decide support vectors(samples with alpha > epsilon are classed as support vectors)

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# returns train data divided by labels and a validation set(used for hyper parameter tuning of C)
def read_data(path):
    train_d = np.genfromtxt(path,delimiter=',')
    #train_df = pd.read_csv(path)
    #train_d = train_df.to_numpy()

    #train_d = train_d[:500]

    np.random.shuffle(train_d)
    n = int(0.8*train_d.shape[0])

    train_d,val_d = train_d[:n][:] , train_d[n:][:]

    val_x,val_y = val_d[:,:-1]/255 , val_d[:,-1]

    train_data = []

    for i in range(k):
        train_data_i = (train_d[np.where(train_d[:,-1] == i)][:,:-1])/255
        train_data.append(train_data_i)

    return (train_data,val_x,val_y)

(train_data,val_x,val_y) = read_data(train_file)
#(test_data,_,_) = read_data(test_file)
#val_data = read_data('val.csv')

def get_test_x(test_path):
    test_d = np.genfromtxt(test_path,delimiter=',')
    #test_df = pd.read_csv(test_path)
    #test_d = test_df.to_numpy()

    #test_d = test_d[:200]

    test_x,test_y = test_d[:,:-1]/255 , test_d[:,-1]

    return (test_x,test_y)

(test_x,test_y) = get_test_x(test_file)

# Converts the data to binary classification between labels i and j
def formulate_data(train_data,i,j):

    # classifying i as 1's and j as -1's
    train_label_i = np.ones((train_data[i].shape[0],1))
    train_label_j = np.full((train_data[j].shape[0],1),-1)

    train_x = np.concatenate((train_data[i],train_data[j]))
    train_y = np.concatenate((train_label_i,train_label_j))

    return (train_x,train_y)


# convert data to a form usable by sklearn
def get_data_OnevsOne_sklearn(train_data):
    train_labels = []

    for i in range(k):
        train_labels.append(np.full((train_data[i].shape[0]),i))

    return (np.concatenate(train_data),np.concatenate(train_labels))


# train a linear svm. returns (w,b) the set of primal objectives
# parameters are the train data, train labels and i and j which are the labels on which the model is being trained
def train_linear_svm(train_x,train_y,i,j,C=1):

    m = train_y.shape[0]

    p1 = train_y*train_x

    P = matrix(np.matmul(p1,p1.T) , tc='d')
    q = matrix(np.full((m,1),-1),tc='d')
    G = matrix(np.concatenate((-np.identity(m),np.identity(m))),tc='d')
    h = matrix(np.concatenate((np.zeros((m,1)),np.full((m,1) , C))),tc='d')
    A = matrix(train_y.T , tc='d')
    b = matrix(np.zeros((1,1)) , tc='d')

    sol = solvers.qp(P,q,G,h,A,b)

    alpha = np.array(sol['x'])

    indices = np.where(alpha > epsilon)

    alpha = alpha[indices,:][0]
    train_x = train_x[indices,:][0] # The set of support vectors
    train_y = train_y[indices,:][0]

    w = np.sum(alpha*train_y*train_x,axis=0)

    w = np.reshape(w , (w.shape[0],1))

    b = (-0.5)*(np.amin(np.matmul(train_x[np.where(train_y == 1),:][0],w)) + np.amax(np.matmul(train_x[np.where(train_y == -1),:][0],w)))

    return (w,b)

# calculate the accuracy of a linear svm
def calc_accuracy_linear(test_x,test_y,w,b):
    
    pred = np.matmul(test_x,w) + b
    return np.mean(pred*test_y >= 0)

# calculates the pairwise distance matrix for a given matrix X
def calc_PairwiseDistance_matrix(train_x):
    diag_dist = np.sum(train_x*train_x,axis=1)
    G = np.matmul(train_x,train_x.T)
    One = np.ones((diag_dist.shape[0],1))

    diag_dist = np.reshape(diag_dist , (diag_dist.shape[0],1))

    return np.matmul(diag_dist,One.T) + np.matmul(One,diag_dist.T) - 2*G

# calculates the Gaussian matrix where G[i][j] = exp(-gamma*||x[i] - x[j]||^2)
def calc_Gaussian_Matrix(train_x,gamma):
    pdist = calc_PairwiseDistance_matrix(train_x)
    return np.exp(-gamma*pdist)

# train a Gaussian kernel svm, returns the dual objective, b and the set of support vectors   
def GaussianKernel_svm(train_x,train_y,C=1):
    m = train_y.shape[0]
    Gaussian = calc_Gaussian_Matrix(train_x,gamma)

    P = matrix((np.matmul(train_y,train_y.T) * Gaussian) , tc='d')
    q = matrix(np.full((m,1),-1),tc='d')
    G = matrix(np.concatenate((-np.identity(m),np.identity(m))),tc='d')
    h = matrix(np.concatenate((np.zeros((m,1)),np.full((m,1) , C))),tc='d')
    A = matrix(train_y.T , tc='d')
    b = matrix(np.zeros((1,1)) , tc='d')

    sol = solvers.qp(P,q,G,h,A,b)

    alpha = np.reshape(np.array(sol['x']) , (m,1))

    indices = np.where(alpha > epsilon)

    alpha = alpha[indices,:][0]
    train_x = train_x[indices,:][0] # The set of support vectors
    train_y = train_y[indices,:][0]
    
    b_helper = np.sum(alpha*train_y*calc_Gaussian_Matrix(train_x,gamma),axis=0)

    b_helper = np.reshape(b_helper , (b_helper.shape[0],1))

    b = (-0.5)*(np.amin(b_helper[np.where(train_y == 1)]) + np.amax(b_helper[np.where(train_y == -1)]))

    return (alpha,b,train_x,train_y)

# predict the labels of test data according to our gaussian svm model
def predict_Gaussian(test_x,gamma,Gaussian_param):
    alpha,b,train_x,train_y = Gaussian_param
    m = train_y.shape[0]
    n = test_x.shape[0]

    diag_dist_train = np.reshape(np.sum(train_x*train_x,axis=1) , (m,1))
    One_train = np.ones((m,1))
    diag_dist_test = np.reshape(np.sum(test_x*test_x,axis=1) , (n,1))
    One_test = np.ones((1,n))

    pdist = np.matmul(diag_dist_train,One_test) + np.matmul(One_train,diag_dist_test.T) - 2*np.matmul(train_x,test_x.T)
    ##print(pdist.shape)

    gaussian = np.exp(-gamma*pdist)
    ##print(gaussian.shape)

    pred = np.sum(alpha*train_y*gaussian,axis=0) + b

    pred = np.reshape(pred , (n,1))

    return pred

def calc_accuracy_Gaussian(Gaussian_param,gamma,test_x,test_y):
    pred = predict_Gaussian(test_x,gamma,Gaussian_param)
    return np.mean(pred*test_y >= 0)

# Train k(k-1)/2 gaussian kernel svm models and return a dictionary of models where model[(i,j)] denotes model trained on labels i and j
def train_OnevsOne(train_data,k,C=1):
    models = {}

    for i in range(k):
        for j in range(i+1,k):
            train_x,train_y = formulate_data(train_data,i,j)
            models[(i,j)] = GaussianKernel_svm(train_x,train_y,C=C)

    return models

def make_prediction_OnevsOne(OnevsOne_model,test_x):
    counts = np.zeros((k,test_x.shape[0],1))

    for i in range(k):
        for j in range(i+1,k):
            Gaussian_param = OnevsOne_model[(i,j)]
            
            pred = predict_Gaussian(test_x,gamma,Gaussian_param)
            
            counts[i] += (pred >= 0)
            counts[j] += (pred < 0)

    return np.argmax(counts,axis=0)

def calc_accuracy_OnevsOne(test_data,OnevsOne_model):
    correct = 0
    total = 0

    confusion_matrix = np.zeros((k,k))

    for i in range(k):
        n = test_data[i].shape[0]
        total += n

        pred = make_prediction_OnevsOne(OnevsOne_model,test_data[i])

        for j in range(k):
            confusion_matrix[j][i] += np.sum(pred == j)

        correct += np.sum(pred == i)

    return (confusion_matrix,correct/total)



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

(train_x,train_y) = get_data_OnevsOne_sklearn(train_data)

def hyper_tuning_sklearn(C_list):

    max_accur = 0
    max_C = 0

    for c in C_list:
        start = time.time()
        model = SVC(decision_function_shape='ovo',gamma=gamma,C=c)
        model.fit(train_x, train_y)
        end = time.time()
        print('Training sklearn ovevsone complete in ' , (end - start) , ' seconds')

        # make predictions
        test_pred = model.predict(val_x)
        score = accuracy_score(val_y, test_pred)
        print(score)

        if score > max_accur:
            max_accur = score
            max_C = c

    return max_C

def perform_kfold(k,C_list):
    cv = KFold(n_splits=k, random_state=1, shuffle=True)
    C_vals = []
    kfold_scores = []
    test_scores = []

    for c in C_list:
        C_vals.append(np.log(c))
        model = SVC(decision_function_shape='ovo',C=c,gamma=gamma)

        start = time.time()
        scores = cross_val_score(model, train_x, train_y, scoring='accuracy', cv=cv, n_jobs=-1)
        end = time.time()
        #print('Training kfold for C = ' , c , ' complete in ' ,  (end - start) , ' seconds')

        kfold_score = np.mean(scores)
        kfold_scores.append(kfold_score*100)

        #print('k fold cross validation accuracy for C = ' , c , ' : ', kfold_score*100 , '%')

        model.fit(train_x,train_y)

        test_pred = model.predict(test_x)
        score = accuracy_score(test_y, test_pred)
        test_scores.append(score*100)
        #print('Test Accuracy for C = ' , c , ' : ' , score*100 , '%')

"""
#print('Training OnevsOne gaussian kernel model')
start = time.time()
OnevsOne_model = train_OnevsOne(train_data,k)
end = time.time()
#print('Training complete in ' , (end - start) , ' seconds')

start = time.time()
test_confusion_matrix,accuracy = calc_accuracy_OnevsOne(test_data,OnevsOne_model)
end = time.time()
#print('Test Accuracy(OnevsOne model): ' , accuracy*100 , '%')
#print('Test confusion matrix:')
#print(test_confusion_matrix)
#print('Test Accuracy calculated in ' , (end - start) , ' seconds')

start = time.time()
val_confusion_matrix,accuracy = calc_accuracy_OnevsOne(val_data,OnevsOne_model)
end = time.time()
#print('Validation Accuracy(OnevsOne model): ' , accuracy*100 , '%')
#print('Validation confusion matrix:')
#print(val_confusion_matrix)
#print('Validation Accuracy calculated in ' , (end - start) , ' seconds')
"""

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

C = hyper_tuning_sklearn([1,5,10])

print(C)

OnevsOne_model = train_OnevsOne(train_data,k,C=C)

predictions = make_prediction_OnevsOne(OnevsOne_model,test_x)

predictions = np.reshape(predictions , predictions.shape[0])

print(np.mean(predictions == test_y))

write_predictions(output_file,predictions)


















