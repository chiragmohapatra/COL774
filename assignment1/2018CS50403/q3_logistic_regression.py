"""

This code is an implementation of logistic regression for a binary classification problem using Newton's Method for convergence

"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd # to read csv files
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

data_dir = sys.argv[1]
out_dir = sys.argv[2]
question_part = sys.argv[3]


# This functions reads data and returns a tuple with normalised x, y, x, mean of x and std dev of x as its elements
def read_data():
    x_data = pd.read_csv(os.path.join(data_dir , 'logisticX.csv'))
    y_data = pd.read_csv(os.path.join(data_dir , 'logisticY.csv'))

    x_array = x_data.to_numpy()
    y = y_data.to_numpy()

    x_mean = np.mean(x_array,axis=0)
    x_std = np.std(x_array,axis=0)
    x = (x_array - x_mean)/x_std #normalize x

    return (x,y,x_array,x_mean,x_std)

# Calculates the j,kth element of the Hessian Matrix
def calc_hessian_element(j,k,train,exp_array,m):
    ans = 0

    for i in range(m):
        foo = exp_array[i]/((1 + exp_array[i])**2)
        ans+=(foo*train[i][j]*train[i][k])

    ans*=(-1)
    return ans

class LogisticRegression:
    """

    To converge using Newton's method, we have to use the Hessian for LL(theta)
    After that the update is decrementing by inverse of Hessian times gradient of LL(theta) for each iteration

    The convergence criteria is that |Theta_j^{t+1} - Theta_j^{t}| <= delta for all j

    """

    def __init__(self,delta=1e-07):
        self.delta = delta

    def fit_model(self,x,y):

        m = x.shape[0] # number of training examples
        n = x.shape[1] # number of features
        n+=1 # to account for intercept

        self.theta = np.zeros((n,1)) # since there are two features of x
        self.last = np.zeros((m,1))
        
        exp_array = np.zeros(m) # for every iteration, stores e^{-(theta.T)xi}
        hypothesis = np.zeros((m,1))
        Hessian = np.zeros((n,n)) # this will be updated on every iteration

        ones_tr = np.ones((m,1))
        train = np.append(ones_tr,x,axis=1)

        while(True):
            for i in range(m):
                exp_array[i] = math.exp((-1)*(self.theta[0][0] + self.theta[1][0]*x[i][0] + self.theta[2][0]*x[i][1]))
                hypothesis[i][0] = 1/(1 + exp_array[i])

            residuals = y - hypothesis

            gradient_vector = np.matmul(train.T,residuals)

            for j in range(n):
                for k in range(n):
                    Hessian[j][k] = calc_hessian_element(j,k,train,exp_array,m)

            Hessian_inverse = np.linalg.inv(Hessian)

            self.theta-=(np.matmul(Hessian_inverse,gradient_vector))
            self.last = hypothesis

            max_grad = np.max(gradient_vector,axis=0)

            if(abs(max_grad) <= self.delta):
                break

        return self


(x,y,x_array,x_mean,x_std) = read_data()

my_model = LogisticRegression()

my_model.fit_model(x,y)


if(question_part == 'a'):

    print('Weights learned by by model are:' , my_model.theta.T)

    with open(os.path.join(out_dir , 'output_3a.txt'), 'w') as f:
        print('Weights learned by by model are:' , my_model.theta.T , file=f)

elif(question_part == 'b'):
    col = y.flatten()

    markers = ["+","o"]
    colors = ['b','g']
    for i, c in enumerate(np.unique(col)):
        plt.scatter(x_array[:,0][col==c],x_array[:,1][col==c],c=colors[i], marker=markers[i], label = 'class label =' + str(c))
    plt.legend()

    x1 = np.linspace(-0.25, 1.5, 100)
    x2 = (-1/my_model.theta[2][0])*(my_model.theta[0][0] + my_model.theta[1][0]*x1)

    x1 = x1*x_mean[0] + x_std[0]
    x2 = x2*x_mean[1] + x_std[1]

    plt.plot(x1,x2,color='Red')
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(os.path.join(out_dir , 'output_3b.png'))
    plt.close()





        
