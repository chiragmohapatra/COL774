import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import math

y_list = []
with open('q4y.dat','r') as file: 
   
    # reading each line     
    for line in file: 
        if(line[:-1] == 'Alaska'):
            y_list.append(0)
        elif(line[:-1] == 'Canada'):
            y_list.append(1)

y = np.array(y_list)

#y = y.reshape((len(y_list),1))

x_list = []
with open('q4x.dat','r') as file: 
   
    # reading each line     
    for line in file:
        i = 0 
        while(line[i] != ' '):
            i+=1
        temp1 = (int)(line[:i])
        i+=2
        temp2 = (int)(line[i:-1])
        x_list.append([temp1,temp2])

x = np.array(x_list)

x_mean = np.mean(x,axis=0)
x_std = np.std(x,axis=0)
x = (x - x_mean)/x_std #normalize x

# Now x and y contain our data with y being an mx1 matrix and x being an mxn matrix where n is number of features

class GaussianDiscriminantAnalysis:
    """
    We already know the equations for the analytically computed parameters of Gaussian Discriminant Analysis

    Attributes
    -----------

    phi: The Bernoulli parameter for distribution of y
    mu0: The mean of the distribution of x|y=0
    sigma0: The covariance matrix of the distribution of x|y=0
    mu1: The mean of the distribution of x|y=1
    sigma1: The covariance matrix of the distribution of x|y=1
    sigma: If sigma0 = sigma1 = sigma for parameter tying

    """

    def fit_model_parameter_tying(self,x,y):
        m = x.shape[0] # number of training examples
        n = x.shape[1] # number of features

        number_of_1s = np.sum(y)

        self.phi = (number_of_1s)/m

        self.mu0 = np.zeros((1,n))
        self.mu1 = np.zeros((1,n))

        for i in range(m):
            if(y[i] == 1):
                self.mu1+=x[i]
            else:
                self.mu0+=x[i]

        self.mu0/=(m - number_of_1s)
        self.mu1/=(number_of_1s)

        self.sigma = np.zeros((n,n))

        for i in range(m):
            if(y[i] == 1):
                self.sigma+=(np.matmul((x[i] - self.mu1).T , x[i] - self.mu1))
            else:
                self.sigma+=(np.matmul((x[i] - self.mu0).T , x[i] - self.mu0))

        self.sigma/=m

        return self

    #assertion: model is already trained with parameter tying
    def decision_boundary_tying(self):
        sigma_inv = np.linalg.inv(self.sigma)

        #print(sigma_inv.shape)
        #print(self.mu0.shape)

        param2 = np.matmul(sigma_inv,self.mu1.T) - np.matmul(sigma_inv,self.mu0.T)

        param3 = 0.5*(np.matmul(self.mu1 , np.matmul(sigma_inv,self.mu1.T)) - np.matmul(self.mu0 , np.matmul(sigma_inv,self.mu0.T)))

        param4 = math.log(self.phi/(1 - self.phi))


        coeff_x1 = -param2[0][0]
        coeff_x2 = -param2[1][0]

        intercept = param3[0] - param4


        return (coeff_x1,coeff_x2,intercept)

        # The first term in the tuple is an array of the coefficients of all features of x while the second is the intercept term

    def fit_model_general(self,x,y):
        m = x.shape[0] # number of training examples
        n = x.shape[1] # number of features

        number_of_1s = np.sum(y)

        self.phi = (number_of_1s)/m

        self.mu0 = np.zeros((1,n))
        self.mu1 = np.zeros((1,n))

        for i in range(m):
            if(y[i] == 1):
                self.mu1+=x[i]
            else:
                self.mu0+=x[i]

        self.mu0/=(m - number_of_1s)
        self.mu1/=(number_of_1s)

        self.sigma0 = np.zeros((n,n))
        self.sigma1 = np.zeros((n,n))

        for i in range(m):
            if(y[i] == 1):
                self.sigma1+=(np.matmul((x[i] - self.mu1).T , x[i] - self.mu1))
            else:
                self.sigma0+=(np.matmul((x[i] - self.mu0).T , x[i] - self.mu0))

        self.sigma1/=(number_of_1s)
        self.sigma0/=(m - number_of_1s)

        return self

    # assertion: model is already trained with general gda
    def decision_boundary_gen(self):
        sigma0_inv = np.linalg.inv(self.sigma0)
        sigma1_inv = np.linalg.inv(self.sigma1)

        param1 = 0.5*(sigma1_inv - sigma0_inv)

        param2 = np.matmul(sigma1_inv,self.mu1.T) - np.matmul(sigma0_inv,self.mu0.T)

        param3 = 0.5*(np.matmul(self.mu1 , np.matmul(sigma1_inv,self.mu1.T)) - np.matmul(self.mu0 , np.matmul(sigma0_inv,self.mu0.T)))

        param4 = math.log(self.phi/(1 - self.phi)) + 0.5*math.log((np.linalg.det(self.sigma0))/(np.linalg.det(self.sigma1)))

        # the quadratic decision boundary, if x has two features (x1,x2) then:

        coeff_x12 = param1[0][0] # coeff of x1^2
        coeff_x22 = param1[1][1]
        coeff_x1x2 = param1[0][1] + param1[1][0]

        coeff_x1 = -param2[0][0]
        coeff_x2 = -param2[1][0]

        intercept = param3[0] - param4

        return (coeff_x12,coeff_x22,coeff_x1x2,coeff_x1,coeff_x2,intercept)

my_model_tied = GaussianDiscriminantAnalysis()

my_model_tied.fit_model_parameter_tying(x,y)

print('These are the parameters learnt on training model assuming parameter tying:')
print('phi = ' , my_model_tied.phi)

print('mu0 = ' , my_model_tied.mu0)

print('mu1 = ', my_model_tied.mu1)

print('sigma = ', my_model_tied.sigma)

labels = ['Alaska','Canada']
markers = ["+","o"]
colors = ['b','g']
for i, c in enumerate(np.unique(y)):
    plt.scatter(x[:,0][y==c],x[:,1][y==c],c=colors[i], marker=markers[i] , label = labels[i])
plt.legend()
plt.show()
plt.close()

# The following code plots the linear decision boundary for gda
coeff_linear = my_model_tied.decision_boundary_tying()

markers = ["+","o"]
colors = ['b','g']
for i, c in enumerate(np.unique(y)):
    plt.scatter(x[:,0][y==c],x[:,1][y==c],c=colors[i], marker=markers[i] , label = labels[i])
plt.legend()
x1 = np.linspace(-3.0, 3.0, 500)
x2 = np.linspace(-3.0, 3.0, 500)
X, Y = np.meshgrid(x1,x2)
F = coeff_linear[0]*X + coeff_linear[1]*Y + coeff_linear[2]
plt.contour(X,Y,F,[0],colors='Blue')
plt.show()
plt.close()

print('-----------------------------------------\n\n')

my_model_gen = GaussianDiscriminantAnalysis()

my_model_gen.fit_model_general(x,y)

print('These are the parameters learnt upon training model with general gda')

print('phi = ', my_model_gen.phi)
print('mu0 = ', my_model_gen.mu0)
print('mu1 = ', my_model_gen.mu1)
print('sigma0 = ', my_model_gen.sigma0)
print('sigma1 = ', my_model_gen.sigma1)

coeff = my_model_gen.decision_boundary_gen()

markers = ["+","o"]
colors = ['b','g']
for i, c in enumerate(np.unique(y)):
    plt.scatter(x[:,0][y==c],x[:,1][y==c],c=colors[i], marker=markers[i] , label = labels[i])
plt.legend()

x1 = np.linspace(-3.0, 3.0, 500)
x2 = np.linspace(-3.0, 3.0, 500)
X, Y = np.meshgrid(x1,x2)
F = coeff[0]*(X**2) + coeff[1]*(Y**2) + coeff[2]*(X*Y) + coeff[3]*X + coeff[4]*Y + coeff[5]
plt.contour(X,Y,F,[0],colors='Red')
F1 = coeff_linear[0]*X + coeff_linear[1]*Y + coeff_linear[2]
plt.contour(X,Y,F1,[0],colors='Blue')
plt.show()
plt.close()

#print(coeff_linear)
#print(coeff)






