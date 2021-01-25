import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import sys

from sklearn.ensemble import RandomForestClassifier

from queue import Queue

def read_data(file):
    data_df = pd.read_csv(file)
    data_np = data_df.to_numpy(dtype=int)

    return data_np

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

train_data = read_data(sys.argv[2])
test_data = read_data(sys.argv[4])
val_data = read_data(sys.argv[3])

output_file = sys.argv[5]
question = int(sys.argv[1])

label_values = [1,2,3,4,5,6,7] # set according to the given dataset(cover type which we have to predict)
num_attributes = 54 # the number of attributes in our dataset

max_depth = 50

number_of_nodes = 0

# This is simplified as a binary tree
class DTreeNode:

    """
    Attributes of class DTreeNode:

    val: the value predicted by the node

    attribute: if an internal node, then the attribute on which it is split to left and right
    attribute_value: the value of the attribute on which the node is split # the left subtree indicates <= attribute_value and the right indicates > attribute value 
    left: the left child
    right: the right child

    """

    def __init__(self,value,data=None):
        global number_of_nodes
        number_of_nodes += 1
        self.node_id = number_of_nodes

        self.val = value

        self.left = None
        self.right = None

        self.data = data

        self.attribute = None
        self.attribute_value = None


# given the index of the attribute, returns if it is boolean or not
def is_boolean(attribute):
    # decided based on the chosen dataset
    if attribute <= 9:
        return False

    else:
        return True

# split the data
def get_split(data,attr,attr_val):

    left_half = data[np.where(data[:,attr] <= attr_val)]
    right_half = data[np.where(data[:,attr] > attr_val)]

    return (left_half,right_half)

#given an np array of data, calculate the entropy
def calc_entropy(data):
    entropy = 0
    total_size = data.shape[0]

    if total_size == 0:
        return 0

    for label in label_values:
        num_samples = data[np.where(data[:,-1] == label)].shape[0]
        prob = num_samples/total_size

        if prob != 0:
            entropy -= (prob * math.log2(prob))

    return total_size*entropy

# returns the best attribute to splt the data on
def ChooseBestAttribute(data):
    min_entropy = float('inf')
    best_attr = -1
    best_value = -1

    medians = np.median(data,axis=0)

    for attr in range(num_attributes):
        attr_val = medians[attr]

        # if a boolean valued attribute, then we split on 0.5
        if is_boolean(attr):
            attr_val = 0.5

        (left_half,right_half) = get_split(data,attr,attr_val)

        entropy_left = calc_entropy(left_half)

        entropy_right = calc_entropy(right_half)

        entropy = entropy_left + entropy_right

        if entropy < min_entropy:
            min_entropy = entropy
            best_attr = attr
            best_value = attr_val

    return (best_attr,best_value)

def predict(DTree,data):
    pred = []

    for t in data:
        root = DTree

        while True:
            if root.left == None and root.right == None:
                break

            else:

                if t[root.attribute] <= root.attribute_value:
                    root = root.left

                else:
                    root = root.right

        pred.append(root.val)

    return pred


def calc_accuracy(DTree,data):
    correct = 0
    total = data.shape[0]

    for t in data:
        root = DTree

        while True:
            if root.left == None and root.right == None:
                break

            else:

                if t[root.attribute] <= root.attribute_value:
                    root = root.left

                else:
                    root = root.right

        pred = root.val

        if pred == t[-1]:
            correct += 1

    return (correct/total)


def plot_accuracy_graph(number_nodes,acc,label):
    plt.plot(number_nodes ,acc)   
    plt.xlabel('number of nodes')
    plt.ylabel(label)
    plt.show()


# a breadth first search implementation to grow the decision tree
def GrowTree(data,max_depth,plot=False):
    majority_class = np.bincount(data[:,-1]).argmax()

    if data.size == 0:
        return DTreeNode(value=1)

    y = data[:,-1]

    for l in label_values:
        if np.all(y == l):
            return DTreeNode(value=l)

    new_node = DTreeNode(value=np.bincount(y).argmax(),data=data)

    queue = Queue()
    queue.put((new_node,0))

    if plot:
        plotted_depth = 0
        number_nodes = []
        train_acc = []
        test_acc = []
        val_acc = []

    while(not queue.empty()):
        (node,depth) = queue.get()

        if plot:
            if depth == plotted_depth:
                plotted_depth += 1
                number_nodes.append(number_of_nodes)
                train_acc.append(calc_accuracy(new_node,train_data))
                test_acc.append(calc_accuracy(new_node,test_data))
                val_acc.append(calc_accuracy(new_node,val_data))


        if depth == max_depth:
            node.data = None
            continue

        (best_attr,best_value) = ChooseBestAttribute(node.data) # returns the index of the best attribute value

        node.attribute = best_attr
        node.attribute_value = best_value

        (left_half,right_half) = get_split(node.data,best_attr,best_value)
        node.data = None

        if left_half.size == 0:
            node.left = DTreeNode(value=majority_class)

        else:
            y_left = left_half[:,-1]

            for l in label_values:
                if np.all(y_left == l):
                    node.left = DTreeNode(value=l)


            if node.left == None:
                node.left = DTreeNode(value=np.bincount(y_left).argmax(),data=left_half)
                queue.put((node.left,depth+1))


        if right_half.size == 0:
            node.right = DTreeNode(value=majority_class)

        else:
            y_right = right_half[:,-1]

            for l in label_values:
                if np.all(y_right == l):
                    node.right = DTreeNode(value=l)

            if node.right == None:
                node.right = DTreeNode(value=np.bincount(y_right).argmax(),data=right_half)
                queue.put((node.right,depth+1))


    if plot:
        plot_accuracy_graph(number_nodes,train_acc,'train accuracy')
        plot_accuracy_graph(number_nodes,test_acc,'test accuracy')
        plot_accuracy_graph(number_nodes,val_acc,'validation accuracy')
        
    return new_node

"""
start = time.time()
DTree = GrowTree(train_data,max_depth,plot=False)
end = time.time()

print('Training complete in: ' , end - start , 'seconds')

print(number_of_nodes)
"""

number_of_prunings = 0

def get_size(node):
    if node == None:
        return 0

    elif node.left == None and node.right == None:
        return 1

    else:
        return 1 + get_size(node.left) + get_size(node.right)

"""
number_nodes = []
train_acc = []
test_acc = []
val_acc = []
"""

def pruning(data,node):
    global number_of_prunings

    node_error = np.count_nonzero(data[:,-1] != node.val)
    left_error = 0
    right_error = 0

    if node.left == None and node.right == None:
        return node_error

    (left_half,right_half) = get_split(data,node.attribute,node.attribute_value)

    if node.left != None and left_half.size > 0:
        left_error = pruning(left_half,node.left)

    if node.right != None and right_half.size > 0:
        right_error = pruning(right_half,node.right)

    # If the error in the children is greater, then we will prune them out

    if (left_error + right_error > node_error):
        node.left = None
        node.right = None

        number_of_prunings += 1

        """
        if number_of_prunings % 100 == 0:
            number_nodes.append(get_size(DTree))
            train_acc.append(calc_accuracy(DTree,train_data))
            test_acc.append(calc_accuracy(DTree,test_data))
            val_acc.append(calc_accuracy(DTree,val_data))
        """
                
        return node_error

    return left_error + right_error

"""
pruning(val_data,DTree)
print(number_of_prunings)
"""

"""
plot_accuracy_graph(number_nodes,train_acc,'train accuracy')
plot_accuracy_graph(number_nodes,test_acc,'test accuracy')
plot_accuracy_graph(number_nodes,val_acc,'validation accuracy')
"""

################## Autograding part ##################

DTree = GrowTree(train_data,max_depth,plot=False)

if question == 2:
    pruning(val_data,DTree)

pred = predict(DTree,test_data)

write_predictions(output_file, pred)


################################1c#################################
# I have not used GridSearchCV since it uses cross validation which is not required for our case

"""
train_x,train_y = train_data[:,:-1] , train_data[:,-1]
test_x,test_y = test_data[:,:-1] , test_data[:,-1]
val_x,val_y = val_data[:,:-1] , val_data[:,-1]

n_estimators_list = [50,150,250,350,450]
max_features_list = [0.1,0.3,0.5,0.7,0.9]
min_samples_split_list = [2,4,6,8,10]

best_oob_score = -1
best_n_estimators = -1
best_max_features = -1
best_min_samples_split = -1

for n_estimators in n_estimators_list:
    for max_features in max_features_list:
        for min_samples_split in min_samples_split_list:
            clf = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features, min_samples_split = min_samples_split, oob_score=True)
            clf.fit(train_x,train_y)
            oob_score = clf.oob_score_
            if oob_score > best_oob_score:
                best_oob_score = oob_score
                best_n_estimators = n_estimators
                best_max_features = max_features
                best_min_samples_split = min_samples_split

print(best_oob_score)
print(best_n_estimators)
print(best_max_features)
print(best_min_samples_split)

"""


#####################1d########################

"""
train_x,train_y = train_data[:,:-1] , train_data[:,-1]
test_x,test_y = test_data[:,:-1] , test_data[:,-1]
val_x,val_y = val_data[:,:-1] , val_data[:,-1]

best_n_estimators = 450
best_max_features = 0.7
best_min_samples_split = 2

n_estimators_list = [50,150,250,350,450]
max_features_list = [0.1,0.3,0.5,0.7,0.9]
min_samples_split_list = [2,4,6,8,10]

test_acc_estimators = []
test_acc_features = []
test_acc_samples_split = []

test_acc_estimators = []
test_acc_features = []
test_acc_samples_split = []

val_acc_estimators = []
val_acc_features = []
val_acc_samples_split = []

for n_estimators in n_estimators_list:
    clf = RandomForestClassifier(n_estimators = n_estimators, max_features = 0.7, min_samples_split = 2, oob_score=True)
    clf.fit(train_x,train_y)
    test_acc_estimators.append(clf.score(test_x,test_y))
    val_acc_estimators.append(clf.score(val_x,val_y))

for max_features in max_features_list:
    clf = RandomForestClassifier(n_estimators = 450, max_features = max_features, min_samples_split = 2, oob_score=True)
    clf.fit(train_x,train_y)
    test_acc_features.append(clf.score(test_x,test_y))
    val_acc_features.append(clf.score(val_x,val_y))

for min_samples_split in min_samples_split_list:
    clf = RandomForestClassifier(n_estimators = 450, max_features = 0.7, min_samples_split = min_samples_split, oob_score=True)
    clf.fit(train_x,train_y)
    test_acc_samples_split.append(clf.score(test_x,test_y))
    val_acc_samples_split.append(clf.score(val_x,val_y))

print(test_acc_estimators)
print(test_acc_features)
print(test_acc_samples_split)

print(val_acc_estimators)
print(val_acc_features)
print(val_acc_samples_split)

"""





