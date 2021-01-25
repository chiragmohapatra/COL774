import matplotlib
matplotlib.use('Agg')

import math
import numpy as np
import pandas as pd # to read csv
import matplotlib.pyplot as plt
import time

m = 1000000 # the number of training samples
x1 = np.random.normal(3,2,m)
x2 = np.random.normal(-1,2,m)
noises = np.random.normal(0,math.sqrt(2),m)

x1 = x1.reshape((m,1))
x2 = x2.reshape((m,1))
noises = noises.reshape((m,1))

theta = np.array([[3],[1],[2]])

y = theta[0] + theta[1]*x1 + theta[2]*x2 + noises # samples are created

x = np.concatenate((x1,x2),axis=1)

ones_arr = np.ones((m,1))

train = np.hstack((ones_arr,x,y))

def compute_error(test,theta):
    hypothesis = np.matmul(test[:,:3],theta)

    residuals = hypothesis - test[:,3:4]

    return np.sum((residuals)**2)/(2*test.shape[0])



class SGD:
    """
    This is linear regression using Stochastic Gradient Descent.

    Parameters
    ----------
    
    eta: Learning rate
    b: Batch size
    epsilon: Stopping parameter


    Atrributes
    ----------

    thetas: The weights learnt from the model

    """

    def __init__(self,eta,b,epsilon):
        self.eta = eta
        self.b = b
        self.epsilon = epsilon

    def fit_model(self,train):
        """
        Function parameters:

        """
        start = time.time()

        m = x.shape[0] # number of training examples
        n = x.shape[1] # number of features
        n+=1 # to account for intercept

        self.theta = np.zeros((n,1)) # assuming n features of x
        b = self.b

        if(b >= m):
            b = m

        self.theta0_list = []
        self.theta1_list = []
        self.theta2_list = []

        self.costs = []
        self.costs.append(0)
        epochs = 0

        cont = True
        iter = 0
        k = 0

        while(cont):
            #np.random.shuffle(train)
            cost_ = 0
            iterations = 0

            for i in range(m//b):
                x_i = train[i*b:(i+1)*b,:3]

                residuals = np.matmul(x_i,self.theta) - train[i*b:(i+1)*b,3:4]

                gradient_vector = np.matmul(x_i.T,residuals)

                self.theta0_list.append(self.theta[0][0])
                self.theta1_list.append(self.theta[1][0])
                self.theta2_list.append(self.theta[2][0])

                self.theta-=(self.eta/b)*gradient_vector
                iter+=1

                cost_+=np.sum((residuals)**2)/(2*b)

                iterations+=1

                if(b <= 100):
                    if(iterations == 1000):
                        cost_/=1000
                        self.costs.append(cost_)
                        iterations = 0
                        cost_ = 0

                        if(abs(self.costs[-1] - self.costs[-2]) <= self.epsilon):
                            cont = False
                            break

                elif(b <= 10000):
                    self.costs.append(cost_)
                    cost_ = 0
                    iterations = 0
                    if(abs(self.costs[-1] - self.costs[-2]) <= self.epsilon):
                        k+=1
                        if(k == 200):
                            cont = False
                            break

                else:
                    self.costs.append(cost_)
                    cost_ = 0
                    if(abs(self.costs[-1] - self.costs[-2]) <= self.epsilon):
                        cont = False
                        break
                    

            epochs+=1

        end = time.time()
        print('Weights learned:' , self.theta)
        print('Time taken:' , end - start , 'seconds')
        print('Number of iterations:' , iter)
        print('Least square error:' , compute_error(train,self.theta))

        return self

print('Training For batch size 1:')
my_model1 = SGD(0.001,1,0.001)

my_model1.fit_model(train)
print('------------------------')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(my_model1.theta0_list,my_model1.theta1_list,my_model1.theta2_list)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('theta2')
ax.set_title('For batch size 1')
plt.savefig('batch_size1.png')

#print(my_model1.theta)

print('Training For batch size 100:')
my_model2 = SGD(0.001,100,0.0025)

my_model2.fit_model(train)
print('------------------------')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(my_model2.theta0_list,my_model2.theta1_list,my_model2.theta2_list)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('theta2')
ax.set_title('For batch size 100')
plt.savefig('batch_size100.png')

#print(my_model2.theta)

print('Training For batch size 10000:')
my_model3 = SGD(0.001,10000,0.001)

my_model3.fit_model(train)

print('------------------------')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(my_model3.theta0_list,my_model3.theta1_list,my_model3.theta2_list)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('theta2')
ax.set_title('For batch size 10000')
plt.savefig('batch_size10000.png')

#print(my_model2.theta)

print('Training For batch size 1000000:')
my_model4 = SGD(0.001,1000000,1e-05)

my_model4.fit_model(train)
print('------------------------')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(my_model4.theta0_list,my_model4.theta1_list,my_model4.theta2_list)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('theta2')
ax.set_title('For batch size 1000000')
plt.savefig('batch_size1000000.png')

test = pd.read_csv('q2test.csv')

test = test.to_numpy()

ones_arr_test = np.ones((test.shape[0],1))

test = np.hstack((ones_arr_test,test))
print('------------------------')
print('------------------------')
print('------------------------')

print('Test error:' , compute_error(test,theta))

print('Running test data on batch size 1:')
print('Error:' , compute_error(test,my_model1.theta))
print('------------------------------------')

print('Running test data on batch size 100:')
print('Error:' , compute_error(test,my_model2.theta))
print('------------------------------------')

print('Running test data on batch size 10000:')
print('Error:' , compute_error(test,my_model3.theta))
print('------------------------------------')

print('Running test data on batch size 1000000:')
print('Error:' , compute_error(test,my_model4.theta))
print('------------------------------------')












#print(my_model.costs)
