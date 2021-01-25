import sys
import os
import numpy as np


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def run(x_train, y_train, x_test, batch_size, hidden_layer_list, activation):
    num_classes = 5

    # Cool Stuff
    
    predictions = [1 ,0, 1]
    return predictions


def main():
    x_train = sys.argv[1]
    y_train = sys.argv[2]
    x_test = sys.argv[3]
    output_file = sys.argv[4]
    batch_size = int(sys.argv[5])
    hidden_layer_list = [int(i) for i in sys.argv[6].split()]
    activation = sys.argv[7]

    output = run(x_train, y_train, x_test, batch_size, hidden_layer_list, activation)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
