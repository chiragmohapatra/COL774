import matplotlib
matplotlib.use('Agg')

import pandas as pd # to read csv files
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

data_dir = sys.argv[1]
out_dir = sys.argv[2]
question_part = sys.argv[3]

# This functions reads data and returns a tuple with normalised x, y and x as its elements
def read_data():
    x_data = pd.read_csv(os.path.join(data_dir,'linearX.csv'))
    y_data = pd.read_csv(os.path.join(data_dir,'linearY.csv'))

    x_array = x_data.to_numpy()
    y = y_data.to_numpy()

    x_mean = np.mean(x_array,axis=0)
    x_std = np.std(x_array,axis=0)
    x = (x_array - x_mean)/x_std #normalize x

    return (x,y,x_array)


class LinearRegression:
    """
    
    Parameters
    ----------
    
    eta: Learning rate
    epsilon: Stopping parameter |J(theta^{t+1}) - J(theta^{t})| <= epsilon
    
    Learned attributes
    ------------------
    
    theta: the weights learnt by the model finally
    costs: an array of the value of the cost function at each iteration
    weights: an array of the weights at each iteration
    
    """

    def __init__(self,eta=0.01,epsilon=1e-10):
        self.eta = eta
        self.epsilon = epsilon

    def fit_model(self,x,y):

        m = x.shape[0] # number of training examples
        n = x.shape[1] # number of features
        n+=1 # to account for intercept

        self.theta = np.zeros((n,1)) # assuming n features of x
        self.costs = []
        self.weight0 = []
        self.weight1 = []

        ones_tr = np.ones((m,1))
        train = np.append(ones_tr,x,axis=1)

        epochs = 0

        while(True):
            residuals = np.matmul(train,self.theta) - y

            gradient_vector = np.matmul(train.T,residuals)

            self.weight0.append(self.theta[0][0])
            self.weight1.append(self.theta[1][0])

            self.theta-=(self.eta/m)*gradient_vector

            cost = np.sum((residuals)**2)/(2*m)

            self.costs.append(cost)

            epochs+=1

            if(epochs > 2):
                if(abs(self.costs[-1] - self.costs[-2]) <= self.epsilon): # the stopping criteria
                    break

        return self


def calc_J(train , y , theta , m):
    hypothesis = np.matmul(train,theta)

    residuals = hypothesis - y

    return np.sum((residuals)**2)/(2*m)


# the following is the code for the 3D plot 
def plot_3D_mesh(my_model,x,y):
    m = x.shape[0]
    ones_tr = np.ones((m,1))
    train = np.append(ones_tr,x,axis=1)

    the0 = np.linspace(-0.3,1.8,60)
    the1 = np.linspace(-1,1,60)

    the0_mesh , the1_mesh = np.meshgrid(the0,the1)

    z = np.array([
            calc_J(train , y , np.array([[a] , [b]]) , m)
            for a,b in zip(np.ravel(the0_mesh) , np.ravel(the1_mesh))
        ])

    Z = z.reshape((the0_mesh.shape[0] , the1_mesh.shape[0]))

    for i in range(0,800,200):
        fig = plt.figure(2)
        sub_fig = fig.add_subplot(111 , projection='3d')
        sub_fig.plot(my_model.weight0[:i] , my_model.weight1[:i] , my_model.costs[:i] , color = 'r' , alpha=0.5)
        surface = sub_fig.plot_surface(the0_mesh, the1_mesh, Z, rstride=1, cstride=1, color='b', alpha=0.5)
        red_part = mpatches.Patch(color='blue', label='J(theta)')
        blue_part = mpatches.Patch(color='red' , label='Path followed by the batch gradient descent')
        sub_fig.set_xlabel('theta0')
        sub_fig.set_ylabel('theta1')
        sub_fig.set_zlabel('J(theta)')
        plt.legend(loc='upper right', handles=[red_part,blue_part])
        sub_fig.set_title('J(theta) vs theta')
        #plt.pause(0.05)
    plt.savefig(os.path.join(out_dir,'output_1c.png'))
    plt.close()


# given the trained model, prints contours
def plot_contours(my_model,x,y):
    m = x.shape[0]
    ones_tr = np.ones((m,1))
    train = np.append(ones_tr,x,axis=1)

    string1 = ""
    string2 = ""

    if(my_model.eta == 0.01):
        string1 = os.path.join(out_dir,'output_1d.png')
        string2 = 'contours'
    else:
        string1 = os.path.join(out_dir , 'output_1e_eta=' + str(my_model.eta) + '.png')
        string2 = 'contours for eta=' + str(my_model.eta)


    fig = plt.figure()
    the0 = np.linspace(-0.3,1.8,60)
    the1 = np.linspace(-1,1,60)

    the0_mesh , the1_mesh = np.meshgrid(the0,the1)

    z = np.array([
            calc_J(train , y , np.array([[a] , [b]]) , m)
            for a,b in zip(np.ravel(the0_mesh) , np.ravel(the1_mesh))
        ])

    Z = z.reshape((the0_mesh.shape[0] , the1_mesh.shape[0]))

    cnt = 400
    if(my_model.eta == 0.001):
        cnt = 2500

    for i in range(0,cnt,100):
        plt.title(string2)
        plt.xlabel('theta0')
        plt.ylabel('theta1')
        plt.plot(my_model.weight0[:20] , my_model.weight1[:20])
        plt.contour(the0_mesh,the1_mesh,Z,20,cmap="RdBu",zorder=1)
        plt.plot(my_model.weight0[:i],my_model.weight1[:i])
        #plt.pause(0.05)
    plt.savefig(string1)
    plt.close()


# This function plots contours for different learning rates with the same stopping criteria
def check_diff_learning(eta):
    my_model1 = LinearRegression(eta)

    my_model1.fit_model(x,y)

    plot_contours(my_model1,x,y)


(x,y,x_array) = read_data()

my_model = LinearRegression()
my_model.fit_model(x,y)


# Here starts the different parts computation
if(question_part == 'a'):
    
    print('Learning rate eta:' , my_model.eta)
    print('Stopping parameter epsilon:' , my_model.epsilon)
    print('Number of epochs required to converge:' , len(my_model.costs))
    print('Weights learned by model:' , my_model.theta.T)

    with open(os.path.join(out_dir,'output_1a.txt'), 'w') as f:
        print('Learning rate eta:' , my_model.eta , file=f)
        print('Stopping parameter epsilon:' , my_model.epsilon , file=f)
        print('Number of epochs required to converge:' , len(my_model.costs) , file=f)
        print('Weights learned by model:' , my_model.theta.T , file=f)

elif(question_part == 'b'):
    
    # The following code plots our hypothesis and data points
    plt.scatter(x_array,y)
    y_pred = my_model.theta[0] + my_model.theta[1]*x
    plt.plot(x_array,y_pred,'-r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(out_dir , 'output_1b.png'))
    plt.close()

elif(question_part == 'c'):
    plot_3D_mesh(my_model,x,y)

elif(question_part == 'd'):
    plot_contours(my_model,x,y)

elif(question_part == 'e'):
    # plotting contours for different learning rates
    check_diff_learning(0.001)

    check_diff_learning(0.025)

    check_diff_learning(0.1)
















