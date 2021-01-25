import sys
import os
import numpy as np


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def run(train_data, test_data):
    num_classes = 5
    predictions = []
    with open(test_data, 'r') as fp:
        for _ in fp:
            p = np.random.randint(low=1, high=num_classes+1, size=1)
            predictions.append(p[0])
    return predictions


def main():
    question = sys.argv[1]
    train_data = sys.argv[2]
    val_data = sys.argv[3]
    test_data = sys.argv[4]
    output_file = sys.argv[5]
    output = run(train_data, test_data)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
