import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

def read_data(file):
    data_df = pd.read_csv(file)
    data_np = data_df.to_numpy(dtype=int)

    return data_np

    """
    x_data , y_data = data_np[:,:-1] , data_np[:.-1]

    return (x_data,y_data)
    """

train_data = read_data('train.csv')
test_data = read_data('test.csv')
val_data = read_data('val.csv')

label_values = [1,2,3,4,5,6,7] # set according to the given dataset(cover type which we have to predict)
num_attributes = 54 # the number of attributes in our dataset

max_depth = 25

number_of_nodes = 0

# This is simplified as a binary tree
class DTreeNode:

    """
    Attributes of class DTreeNode:

    isLeaf: (boolean) if true then the node is a leaf
    val: if the node is a leaf, then the value of the leaf

    attribute: if an internal node, then the attribute on which it is split to left and right
    attribute_value: the value of the attribute on which the node is split # the left subtree indicates <= attribute_value and the right indicates > attribute value 
    left: the left child
    right: the right child

    """

    def __init__(self,isLeaf,value,attr=0,attr_val=0):
        global number_of_nodes
        number_of_nodes += 1
        self.node_id = number_of_nodes

        self.isLeaf = isLeaf
        self.val = value

        self.left = None
        self.right = None

        if not isLeaf:
            self.attribute = attr
            self.attribute_value = attr_val


# given the index of the attribute, returns if it is boolean or not
def is_boolean(attribute):
    # decided based on the chosen dataset
    if attribute <= 9:
        return False

    else:
        return True

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

def GrowTree(data,depth):
    global nodes

    if data.size == 0:
        return DTreeNode(isLeaf=True,value=1)

    y = data[:,-1]

    if depth == 0:
        return DTreeNode(isLeaf=True,value=np.bincount(y).argmax())

    for l in label_values:
        if np.all(y == l):
            return DTreeNode(isLeaf=True,value=l)

    (best_attr,best_value) = ChooseBestAttribute(data) # returns the index of the best attribute value

    new_node = DTreeNode(isLeaf=False,value=np.bincount(y).argmax(),attr=best_attr,attr_val=best_value)

    (left_half,right_half) = get_split(data,best_attr,best_value)

    new_node.left = GrowTree(left_half , depth-1, new_node)
    new_node.right = GrowTree(right_half , depth-1, new_node)

    return new_node

start = time.time()
DTree = GrowTree(train_data,max_depth)
end = time.time()

print('Training complete in: ' , end - start , 'seconds')

print(number_of_nodes)

def calc_accuracy(DTree,test_data):
    correct = 0
    total = test_data.shape[0]

    for t in test_data:
        DTree_root = DTree

        while not DTree_root.isLeaf:
            if t[DTree_root.attribute] <= DTree_root.attribute_value:
                DTree_root = DTree_root.left

            else:
                DTree_root = DTree_root.right

        pred = DTree_root.val

        if pred == t[-1]:
            correct += 1

    return (correct/total)

def get_accuracies(data,DTree):
    correct_pred = np.zeros(number_of_nodes , dtype=int)

    majority_class = np.bincount(data[:,-1]).argmax()
    correct_pred[0] = np.count_nonzero(data[:,-1] == majority_class)
    
    for t in data:
        root = DTree

        while not root.isLeaf:
            if t[root.attribute] <= root.attribute_value:
                new_root = root.left
            else:
                new_root = root.right
            correct_pred[root.node_id:new_root.node_id] += (new_root.val == t[-1])
            root = new_root

        correct_pred[root.node_id:] += (root.val == t[-1])

    return correct_pred/data.shape[0]

"""
no_nodes = np.arange(number_of_nodes)
train_acc = get_accuracies(train_data,DTree)
test_acc = get_accuracies(test_data,DTree)
val_acc = get_accuracies(val_data,DTree)

plt.plot(no_nodes ,train_acc , label='train accuracy')
plt.plot(no_nodes ,test_acc , label='test accuracy')
plt.plot(no_nodes ,val_acc , label='validation accuracy')
plt.xlabel('number of nodes')
plt.ylabel('accuracy')
plt.legend()
plt.show()
"""

def pruning(data,node):
    node_error = np.count_nonzero(data[:,-1] != node.val)
    left_error = 0
    right_error = 0

    if node.isLeaf:
        return node_error

    if node.left == None and node.right == None:
        return node_error

    (left_half,right_half) = get_split(data,node.attribute,node.attribute_value)

    if node.left != None and left.shape[0] > 0:
        left_error = pruning(left_half,node.left)

    if node.right != None and right.shape[0] > 0:
        right_error = pruning(right_half,node.right)

    # If the error in the children is greater, then we will prune them out

    if (left_error + right_error > node_error):
        node.left = None
        node.right = None
        return node_error

    return left_error + right_error

"""
def pruning(DTree):
    size_after_pruning = [number_of_nodes]
    train_acc_pruning = [train_acc[-1]]
    test_acc_pruning = [test_acc[-1]]
    val_acc_pruning = [val_acc[-1]]

    stack = []
    stack.append(DTree)

    while(not stack):
        node = stack.pop()

        if node.isLeaf:
            continue

        if val_acc[node.right.node_id] < val_acc[node.node_id]:
            n = node.right.size
            size_after_pruning.append(size_after_pruning[-1] - n)

            val_acc[node.right.node_id + n:] += (val_acc[node.right.node_id + n - 1] - val_acc[node.right.node_id])
            train_acc[node.right.node_id + n:] += (train_acc[node.right.node_id + n - 1] - train_acc[node.right.node_id])
            test_acc[node.right.node_id + n:] += (test_acc[node.right.node_id + n - 1] - test_acc[node.right.node_id])

            train_acc_pruning.append(train_acc[-1])
            test_acc_pruning.append(test_acc[-1])
            val_acc_pruning.append(val_acc[-1])

        else:
            stack.append(node.right)


        if val_acc[node.left.node_id] < val_acc[node.node_id]:
            n = node.left.size
            size_after_pruning.append(size_after_pruning[-1] - n)

            val_acc[node.left.node_id + n:] += (val_acc[node.left.node_id + n - 1] - val_acc[node.left.node_id])
            train_acc[node.left.node_id + n:] += (train_acc[node.left.node_id + n - 1] - train_acc[node.left.node_id])
            test_acc[node.left.node_id + n:] += (test_acc[node.left.node_id + n - 1] - test_acc[node.left.node_id])

            train_acc_pruning.append(train_acc[-1])
            test_acc_pruning.append(test_acc[-1])
            val_acc_pruning.append(val_acc[-1])

        else:
            stack.append(node.left)

    plt.plot(size_after_pruning ,train_acc_pruning , label='train accuracy')
    plt.plot(size_after_pruning ,test_acc_pruning , label='test accuracy')
    plt.plot(size_after_pruning ,val_acc_pruning , label='validation accuracy')
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
"""

    


        

        





