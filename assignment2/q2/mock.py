import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import time

k = 10 # the number of labels 
gamma = 0.05 # parameter for Gaussian Kernel

def read_data(path):

    train_df = pd.read_csv(path)
    train_d = train_df.to_numpy()

    train_data = []

    for i in range(k):
        train_data_i = (train_d[np.where(train_d[:,-1] == i)][:,:-1])/255
        train_data.append(train_data_i)

    return train_data

train_data = read_data('train.csv')
test_data = read_data('test.csv')
val_data = read_data('val.csv')

# Converts the data to binary classification between labels i and j
def formulate_data(train_data,i,j):

    # classifying i as 1's and j as -1's
    train_label_i = np.ones((train_data[i].shape[0],1))
    train_label_j = np.full((train_data[j].shape[0],1),-1)

    train_x = np.concatenate((train_data[i],train_data[j]))
    train_y = np.concatenate((train_label_i,train_label_j))

    return (train_x,train_y)

def train_linear_svm(train_x,train_y,i,j):

    m = train_y.shape[0]

    p1 = train_y*train_x

    P = matrix(np.matmul(p1,p1.T) , tc='d')
    #P = matrix((np.matmul(train_y,train_y.T) * np.matmul(train_x,train_x.T)) , tc='d')
    q = matrix(np.full((m,1),-1),tc='d')
    G = matrix(np.concatenate((-np.identity(m),np.identity(m))),tc='d')
    h = matrix(np.concatenate((np.zeros((m,1)),np.ones((m,1)))),tc='d')
    A = matrix(train_y.T , tc='d')
    b = matrix(np.zeros((1,1)) , tc='d')

    sol = solvers.qp(P,q,G,h,A,b)

    alpha = np.array(sol['x'])

    w = np.sum(alpha*p1,axis=0)

    w = np.reshape(w , (w.shape[0],1))

    b = (-0.5)*(np.amin(np.matmul(train_data[i],w)) + np.amax(np.matmul(train_data[j],w)))

    return (w,b)

def calc_accuracy_linear(test_x,test_y,w,b):
    
    #n = test_x.shape[0]
    #correct = 0

    #print(test_x.shape)
    #print(w.shape)

    """
    for i in range(n):
        pred = 0
        if np.matmul(test_x[i],w) + b >= 0:
            pred = 1
        else:
            pred = -1
        if(pred == test_y[i]):
            correct += 1

    return correct/n
    """
    pred = np.matmul(test_x,w) + b

    return np.mean(pred*test_y >= 0)


(train_x,train_y) = formulate_data(train_data,3,4)
(test_x,test_y) = formulate_data(test_data,3,4)
(val_x,val_y) = formulate_data(val_data,3,4)

"""
print('Training linear svm for labels 3(1) and 4(-1)')
(w,b) = train_linear_svm(train_x,train_y,3,4)

print('Test Accuracy for linear svm: ' , calc_accuracy_linear(test_x,test_y,w,b)*100 , '%')

print('Validation Accuracy for linear svm: ' , calc_accuracy_linear(val_x,val_y,w,b)*100 , '%')
"""

def calc_PairwiseDistance_matrix(train_x):
    diag_dist = np.sum(train_x*train_x,axis=1)
    G = np.matmul(train_x,train_x.T)
    One = np.ones((diag_dist.shape[0],1))

    diag_dist = np.reshape(diag_dist , (diag_dist.shape[0],1))

    return np.matmul(diag_dist,One.T) + np.matmul(One,diag_dist.T) - 2*G

def calc_Gaussian_Matrix(train_x,gamma):
    pdist = calc_PairwiseDistance_matrix(train_x)
    return np.exp(-gamma*pdist)
    
def GaussianKernel_svm(train_x,train_y):
    m = train_y.shape[0]
    Gaussian = calc_Gaussian_Matrix(train_x,0.05)

    P = matrix((np.matmul(train_y,train_y.T) * Gaussian) , tc='d')
    q = matrix(np.full((m,1),-1),tc='d')
    G = matrix(np.concatenate((-np.identity(m),np.identity(m))),tc='d')
    h = matrix(np.concatenate((np.zeros((m,1)),np.ones((m,1)))),tc='d')
    A = matrix(train_y.T , tc='d')
    b = matrix(np.zeros((1,1)) , tc='d')

    sol = solvers.qp(P,q,G,h,A,b)

    alpha = np.reshape(np.array(sol['x']) , (m,1))

    b_helper = np.sum(alpha*train_y*Gaussian,axis=0)

    b_helper = np.reshape(b_helper , (b_helper.shape[0],1))

    b = (-0.5)*(np.amin(b_helper[np.where(train_y == 1)]) + np.amax(b_helper[np.where(train_y == -1)]))

    return (alpha,b)

def calc_accuracy_Gaussian(train_x,train_y,test_x,test_y,alpha,b,gamma):
    m = train_y.shape[0]
    n = test_y.shape[0]
    """
    for i in range(n):
        gaussian = np.reshape(np.exp(-gamma*np.sum((train_x - test_x[i])**2,axis=1)) , (m,1))
        pred = 0

        mat = alpha*train_y*gaussian

        if np.sum(mat , axis=0) + b >= 0:
            pred = 1
        else:
            pred = -1

        if(pred == test_y[i]):
            correct += 1

    return correct/n
    """

    """
    pred_list = []
    test_chunks = np.array_split(test_x,6)

    for i in range(6):
        chunk_size = test_chunks[i].shape[0]
        gaussian_chunk = np.reshape(np.exp(-gamma*np.sum((train_x - test_chunks[i][:,None])**2,axis=2)) , (chunk_size,m,1))
        pred_chunk = (np.sum(alpha*train_y*gaussian_chunk,axis=1) + b)
        pred_list.append(pred_chunk)

    pred = np.concatenate(pred_list)

    return np.mean(pred*test_y >= 0)
    """

    diag_dist_train = np.reshape(np.sum(train_x*train_x,axis=1) , (m,1))
    One_train = np.ones((m,1))
    diag_dist_test = np.reshape(np.sum(test_x*test_x,axis=1) , (n,1))
    One_test = np.ones((1,n))

    pdist = np.matmul(diag_dist_train,One_test) + np.matmul(One_train,diag_dist_test.T) - 2*np.matmul(train_x,test_x.T)
    #print(pdist.shape)

    gaussian = np.exp(-gamma*pdist)
    #print(gaussian.shape)

    pred = np.sum(alpha*train_y*gaussian,axis=0) + b

    pred = np.reshape(pred , (n,1))

    return np.mean(pred*test_y >= 0)


print('Training gaussian kernel svm')
alpha,b = GaussianKernel_svm(train_x,train_y)
#print(alpha)
#print(b)
print('Test Accuracy for gaussian kernel svm: ', calc_accuracy_Gaussian(train_x,train_y,test_x,test_y,alpha,b,gamma)*100 , '%')
print('Validation Accuracy for gaussian kernel svm: ', calc_accuracy_Gaussian(train_x,train_y,val_x,val_y,alpha,b,gamma)*100 , '%')


def train_OnevsOne(train_data,k):
    models = {}

    for i in range(k):
        for j in range(i+1,k):
            train_x,train_y = formulate_data(train_data,i,j)
            models[(i,j)] = GaussianKernel_svm(train_x,train_y)

    return models

def make_prediction_OnevsOne(OnevsOne_model,test_x):
    counts = np.zeros((k,test_x.shape[0]))

    test_chunks = np.array_split(test_x,6)

    for i in range(k):
        for j in range(i+1,k):
            (alpha,b) = OnevsOne_model[(i,j)]
            #train_x,train_y = formulate_data(train_data,i,j)
            train_x,train_y = formulate_data(test_data,i,j)
            m = train_x.shape[0]
            """
            gaussian = np.reshape(np.exp(-gamma*np.sum((train_x - x)**2,axis=1)) , (m,1))
            pred = 0

            mat = alpha*train_y*gaussian

            if np.sum(mat , axis=0) + b >= 0:
                pred = i
            else:
                pred = j

            counts[pred] += 1

            if counts[pred] > max_cnt:
                max_cnt = counts[pred]
                max_pred = pred
            """
            pred_list = []

            for l in range(6):
                chunk_size = test_chunks[l].shape[0]
                gaussian_chunk = np.reshape(np.exp(-gamma*np.sum((train_x - test_chunks[l][:,None])**2,axis=2)) , (chunk_size,m,1))
                pred_chunk = (np.sum(alpha*train_y*gaussian_chunk,axis=1) + b)
                pred_list.append(pred_chunk)

            pred = np.concatenate(pred_list)
            pred = np.reshape(pred , pred.shape[0])

            counts[i] += (pred >= 0)
            counts[j] += (pred < 0)
            """
            i_labels = np.sum(pred >= 0)

            counts[i] += i_labels
            counts[j] += (pred.shape[0] - i_labels)

            if counts[i] > max_cnt:
                max_cnt = counts[i]
                max_pred = i

            if counts[j] > max_cnt:
                max_cnt = counts[j]
                max_pred = j
            """

    return np.argmax(counts,axis=1)

def calc_accuracy_OnevsOne(test_data,OnevsOne_model):
    correct = 0
    total = 0

    for i in range(k):
        n = test_data[i].shape[0]
        total += n

        #for j in range(n):
        pred = make_prediction_OnevsOne(OnevsOne_model,test_data[i])

        correct += np.sum(pred == i)
            #if pred == i:
                #correct += 1

    return (correct/total)

"""
print('Training OnevsOne gaussian kernel model')
start = time.time()
OnevsOne_model = train_OnevsOne(test_data,k)
end = time.time()
print('Training complete in ' , (end - start) , ' seconds')

start = time.time()
print('Validation Accuracy(OnevsOne model): ' , calc_accuracy_OnevsOne(val_data,OnevsOne_model)*100 , '%')
end = time.time()
print('Accuracy calculated in ' , (end - start) , ' seconds')
"""

"""
Test Accuracy(OnevsOne model):  85.05701140228045 %
Test confusion matrix:
[[404.   0.   0.  18.   0.   1.  56.   0.   1.   0.]
 [  0. 484.   0.  10.   1.   0.   1.   0.   0.   0.]
 [  7.   6. 414.   2.  56.   0.  57.   0.   1.   0.]
 [  7.   2.   4. 412.  14.   0.   6.   0.   0.   0.]
 [  0.   0.  26.   6. 365.   0.  20.   0.   0.   0.]
 [  0.   0.   0.   0.   0. 436.   0.  50.   1.   5.]
 [ 71.   7.  43.  42.  52.   0. 345.   0.   3.   0.]
 [  0.   0.   0.   0.   0.   7.   0. 412.   0.   6.]
 [ 10.   1.  13.  10.  12.  44.  15.   4. 494.   3.]
 [  0.   0.   0.   0.   0.  12.   0.  34.   0. 486.]]
Test Accuracy calculated in  106.93183588981628  seconds
Validation Accuracy(OnevsOne model):  85.03401360544217 %
Validation confusion matrix:
[[202.   0.   2.  12.   0.   0.  20.   0.   0.   0.]
 [  2. 240.   0.  10.   2.   0.   0.   0.   0.   0.]
 [  1.   2. 207.   0.  31.   0.  28.   0.   1.   0.]
 [  4.   2.   1. 195.   5.   1.   1.   0.   1.   0.]
 [  0.   0.  13.   6. 185.   0.  11.   0.   0.   0.]
 [  0.   0.   0.   0.   0. 228.   0.  27.   0.   2.]
 [ 37.   3.  14.  23.  21.   0. 186.   0.   1.   0.]
 [  0.   0.   0.   0.   0.   1.   0. 198.   2.   4.]
 [  4.   3.  12.   4.   6.  14.   4.   4. 245.   5.]
 [  0.   0.   0.   0.   0.   6.   0.  21.   0. 239.]]
Validation Accuracy calculated in  55.307005405426025  seconds
"""

"""
Training sklearn ovevsone complete in  102.48986673355103  seconds
Test Accuracy(OnevsOne model) for sklearn:  86.75735147029407 %
Test confusion matrix for sklearn:
[[428   0   7  19   3   0  35   0   7   0]
 [  1 481   5  10   0   0   3   0   0   0]
 [  5   0 393   7  53   0  42   0   0   0]
 [ 14   1   3 457  12   0  11   0   2   0]
 [  0   1  46  19 394   0  38   0   2   0]
 [  0   0   0   0   0 468   0  23   3   6]
 [ 91   0  57   9  42   0 296   0   5   0]
 [  0   0   0   0   0  18   0 467   1  14]
 [  1   0   3   2   0   2   5   3 483   1]
 [  0   0   1   0   0  10   0  19   0 470]]
Validation Accuracy(OnevsOne model) for sklearn:  87.35494197679071 %
Validation confusion matrix for sklearn:
[[213   1   1  11   0   0  24   0   0   0]
 [  0 240   2   7   0   0   1   0   0   0]
 [  4   0 203   3  25   0  11   0   3   0]
 [  6   0   0 227   7   0   8   0   2   0]
 [  0   1  22  10 202   0  14   0   1   0]
 [  0   0   0   1   0 234   0   9   1   5]
 [ 41   0  28   5  20   0 155   0   1   0]
 [  0   0   0   0   0   9   0 231   1   9]
 [  0   1   1   2   0   0   1   2 243   0]
 [  0   0   0   0   0   5   0  10   0 235]]
 """



"""
Training kfold for C =  1e-05  complete in  3349.9268696308136  seconds
k fold cross validation accuracy for C =  1e-05  :  9.293748240349705 %
Test Accuracy for C =  1e-05  :  64.79295859171835 %
Training kfold for C =  0.001  complete in  3325.0164091587067  seconds
k fold cross validation accuracy for C =  0.001  :  26.863178484107586 %
Test Accuracy for C =  0.001  :  64.35287057411482 %
Training kfold for C =  1  complete in  499.1578118801117  seconds
k fold cross validation accuracy for C =  1  :  87.30610456644688 %
Test Accuracy for C =  1  :  86.75735147029407 %
Training kfold for C =  5  complete in  459.9857611656189  seconds
k fold cross validation accuracy for C =  5  :  88.64839297621694 %
Test Accuracy for C =  5  :  88.17763552710542 %
Training kfold for C =  10  complete in  466.16974449157715  seconds
k fold cross validation accuracy for C =  10  :  88.51504976414513 %
Test Accuracy for C =  10  :  88.27765553110622 %
[-11.512925464970229, -6.907755278982137, 0.0, 1.6094379124341003, 2.302585092994046]
[9.293748240349705, 26.863178484107586, 87.30610456644688, 88.64839297621694, 88.51504976414513]
[0.6479295859171834, 0.6435287057411482, 0.8675735147029406, 0.8817763552710542, 0.8827765553110622]
Total time taken:  10571.363602399826  seconds
"""

"""
Training kfold for C =  1e-05  complete in  3338.551198720932  seconds
k fold cross validation accuracy for C =  1e-05  :  9.293748240349705 %
Test Accuracy for C =  1e-05  :  53.39067813562713 %
Training kfold for C =  0.001  complete in  3344.0825679302216  seconds
k fold cross validation accuracy for C =  0.001  :  9.293748240349705 %
Test Accuracy for C =  0.001  :  53.39067813562713 %
Training kfold for C =  1  complete in  1010.8330547809601  seconds
k fold cross validation accuracy for C =  1  :  87.88835049764145 %
Test Accuracy for C =  1  :  88.07761552310463 %
Training kfold for C =  5  complete in  1105.4513666629791  seconds
k fold cross validation accuracy for C =  5  :  88.25726210762885 %
Test Accuracy for C =  5  :  88.27765553110622 %
Training kfold for C =  10  complete in  1109.2650980949402  seconds
k fold cross validation accuracy for C =  10  :  88.23947346323875 %
Test Accuracy for C =  10  :  88.2376475295059 %
"""



"""
Train reading complete
Transformation of data complete in  122.69301629066467  seconds
Training of Naive Bayes complete in  0.638077974319458  seconds
Test Accuracy of Naive Bayes model:  60.56327495176416 %
C =  0.1
Training of SVM complete in  25.967145204544067  seconds
Test Accuracy of SVM model:  67.89587041385603 %
C =  0.5
Training of SVM complete in  51.4346022605896  seconds
Test Accuracy of SVM model:  67.40229438071165 %
C =  1
Training of SVM complete in  83.10052680969238  seconds
Test Accuracy of SVM model:  66.95059752613709 %
tol =  0.01
Training of SGD Classifier complete in  8.994866371154785  seconds
Test Accuracy of SGD model:  63.20764594145889 %
tol =  0.001
Training of SGD Classifier complete in  10.086521625518799  seconds
Test Accuracy of SGD model:  62.9705798770547 %
tol =  0.0001
Training of SGD Classifier complete in  16.057730436325073  seconds
Test Accuracy of SGD model:  63.27196039426255 %
tol =  1e-05
Training of SGD Classifier complete in  33.821150064468384  seconds
Test Accuracy of SGD model:  63.144827173604156 %
"""
