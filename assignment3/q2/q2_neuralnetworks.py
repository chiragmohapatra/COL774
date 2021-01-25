import numpy as np
import time
import math
import matplotlib.pyplot as plt 

from sklearn.neural_network import MLPClassifier

np.random.seed(0)

train_x = np.load('X_train.npy')

train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1]*train_x.shape[2]))

train_y = np.load('y_train.npy')

test_x = np.load('X_test.npy')

test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1]*test_x.shape[2]))

test_y = np.load('y_test.npy')

def getnHotEncoding(tr_y,num_outputs):
    m = tr_y.shape[0]
    nhot = np.zeros((m,num_outputs))
    for i in range(m):
        nhot[i][tr_y[i]] = 1

    return nhot

def sigmoid(Z):
    np.clip(Z,-500,500,out=Z)
    return 1.0 / (1.0 + np.exp(-Z))

def ReLu(Z):
    return np.maximum(0,Z)

def get_derivative(output):
    return output * (1.0 - output)

def relu_derivative(X):
    X[X<=0] = 0
    X[X>0] = 1
    return X

class NeuralNetworks:

    """
    M: the Mini Batch Size for stochastic gradient descent
    n: the number of input features for a sample
    layers: a list containg the number of units in each hidden layer
    output_labels: The set of the different outputs
    eta: the learning rate, eta=-1 denotes an adaptive learning rate 
    """

    def __init__(self,M,num_features,hidden_layers,target_classes,eta=0.001,epochs=100,useRelu=False):
        self.M = M
        self.n = num_features
        self.layers = hidden_layers
        self.num_outputs = target_classes
        self.eta = eta
        self.epochs = epochs
        self.useRelu = useRelu

    def create_network(self):
        np.random.seed(0)
        self.network = []
        scale = 0.01

        for i in range(len(self.layers)):
            layer = {}
            if i == 0:
                layer['weights'] = np.random.randn(self.n + 1 , self.layers[0] + 1) * scale
            else:
                layer['weights'] = np.random.randn(self.layers[i-1] + 1 , self.layers[i] + 1) * scale

            self.network.append(layer)

        output_layer = {}
        output_layer['weights'] = np.random.randn(self.layers[-1] + 1 , self.num_outputs) * scale
        self.network.append(output_layer)

    def forward_propagate(self,input):
        inputs = np.hstack((input,np.ones((input.shape[0],1))))
        for j in range(len(self.network)):
            Z = np.matmul(inputs,self.network[j]['weights'])
            self.network[j]['output'] = sigmoid(Z)
            #inputs = np.hstack((np.ones((layer['output'].shape[0],1)),layer['output']))
            inputs = self.network[j]['output']

    def forward_propagate_relu(self,input):
        inputs = np.hstack((input,np.ones((input.shape[0],1))))
        for j in range(len(self.network) - 1):
            Z = np.matmul(inputs,self.network[j]['weights'])
            self.network[j]['output'] = ReLu(Z)
            #inputs = np.hstack((np.ones((layer['output'].shape[0],1)),layer['output']))
            inputs = self.network[j]['output']
        Z = np.matmul(inputs,self.network[-1]['weights'])
        self.network[-1]['output'] = sigmoid(Z)
        
    def back_propagate(self,expected):
        n = len(self.network)
        for i in reversed(range(n)):
            if i == n-1:
                self.network[i]['delta'] = (expected - self.network[i]['output']) * get_derivative(self.network[i]['output'])
            else:
                error = np.matmul(self.network[i+1]['delta'],self.network[i+1]['weights'].T)
                #print(error.shape , self.network[i]['output'].shape , self.get_derivative(self.network[i]['output']).shape)
                self.network[i]['delta'] = error * get_derivative(self.network[i]['output'])

    def back_propagate_relu(self,expected):
        n = len(self.network)
        for i in reversed(range(n)):
            if i == n-1:
                self.network[i]['delta'] = (expected - self.network[i]['output']) * get_derivative(self.network[i]['output'])
            else:
                error = np.matmul(self.network[i+1]['delta'],self.network[i+1]['weights'].T)
                #print(error.shape , self.network[i]['output'].shape , self.get_derivative(self.network[i]['output']).shape)
                self.network[i]['delta'] = error * relu_derivative(self.network[i]['output'])


    def train_network(self,train_x,tr_y):
        train_y = getnHotEncoding(tr_y,self.num_outputs)

        self.create_network()

        m = train_x.shape[0]
        b = self.M

        J_theta = []
        epsilon = 1e-04

        for epoch in range(self.epochs):
            average_error = 0
            count = 0
            for i in range(m//b):
                count += 1
                x_i = train_x[i*b:(i+1)*b,:]
                y_i = train_y[i*b:(i+1)*b,:]

                if not self.useRelu:
                    self.forward_propagate(x_i)
                    self.back_propagate(y_i)
                else:
                    self.forward_propagate_relu(x_i)
                    self.back_propagate_relu(y_i)

                average_error += np.sum(np.square(y_i - self.network[-1]['output'])) / (2*b)

                inputs = np.hstack((x_i,np.ones((x_i.shape[0],1))))
                for j in range(len(self.network)):
                    update = np.matmul(inputs.T, self.network[j]['delta'])
                    #print(update)
                    if self.eta != -1:
                        self.network[j]['weights'] += (1/b) * self.eta * update
                    else:
                        self.network[j]['weights'] += (1/b) * (0.5 / math.sqrt(epoch + 1)) * update
                    #inputs = np.hstack((np.ones((self.network[j]['output'].shape[0],1)),self.network[j]['output']))
                    inputs = self.network[j]['output']

            average_error /= count
            J_theta.append(average_error)

            if(len(J_theta) >= 2):
                if(abs(J_theta[-1] - J_theta[-2]) < epsilon):
                    break

    def predict(self,test_x):
        if not self.useRelu:
            self.forward_propagate(test_x)
        else:
            self.forward_propagate_relu(test_x)

        return np.argmax(self.network[-1]['output'] , axis=1)


##########################2b##########################
"""
no_hidden_layers = [1,10,50,100,500]
time_taken = []
train_accuracies = []
test_accuracies = []

for n in no_hidden_layers:
    NNet = NeuralNetworks(100,train_x.shape[1],[n],10)

    start = time.time()
    NNet.train_network(train_x,train_y)
    end = time.time()

    time_taken.append(end - start)

    train_accuracies.append(np.mean(train_y == NNet.predict(train_x) ))

    test_accuracies.append(np.mean(test_y == NNet.predict(test_x) ))

def plot_graph_hidden(acc,name):
    plt.plot(no_hidden_layers ,acc)
    plt.xlabel('number of hidden layers')
    plt.ylabel(name)
    plt.show()

print(time_taken)
print(train_accuracies)
print(test_accuracies)

plot_graph_hidden(time_taken,'time taken to train(in seconds)')
plot_graph_hidden(train_accuracies,'train accuracy')
plot_graph_hidden(test_accuracies,'test accuracy')
"""

############################2c##########################
"""
no_hidden_layers = [1,10,50,100,500]
time_taken = []
train_accuracies = []
test_accuracies = []

for n in no_hidden_layers:
    NNet = NeuralNetworks(100,train_x.shape[1],[n],10,eta=-1)

    start = time.time()
    NNet.train_network(train_x,train_y)
    end = time.time()

    time_taken.append(end - start)

    train_accuracies.append(np.mean(train_y == NNet.predict(train_x) ))

    test_accuracies.append(np.mean(test_y == NNet.predict(test_x) ))

def plot_graph_hidden(acc,name):
    plt.plot(no_hidden_layers ,acc)
    plt.xlabel('number of hidden layers')
    plt.ylabel(name)
    plt.show()

print(time_taken)
print(train_accuracies)
print(test_accuracies)

plot_graph_hidden(time_taken,'time taken to train(in seconds)')
plot_graph_hidden(train_accuracies,'train accuracy')
plot_graph_hidden(test_accuracies,'test accuracy')
"""

#######################2d############################
"""
NNet = NeuralNetworks(100,train_x.shape[1],[100,100],10,eta=-1)

start = time.time()
NNet.train_network(train_x,train_y)
end = time.time()

print(end - start)

print(np.mean(train_y == NNet.predict(train_x) ))

print(np.mean(test_y == NNet.predict(test_x) ))

NNetr = NeuralNetworks(100,train_x.shape[1],[100,100],10,eta=-1,useRelu=True)

start = time.time()
NNetr.train_network(train_x,train_y)
end = time.time()

print(end - start)

print(np.mean(train_y == NNetr.predict(train_x)))

print(np.mean(test_y == NNetr.predict(test_x)))
"""


##########################2e#########################
"""
start = time.time()
clf = MLPClassifier(hidden_layer_sizes = (100,100),random_state=1, max_iter=300 , solver='sgd' , learning_rate='adaptive').fit(train_x, train_y)
end = time.time()
print(end - start)
print(clf.score(train_x,train_y))
print(clf.score(test_x,test_y))
"""




