#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import logging

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

VOCAB_SIZE = 8520
TRAIN_SIZE = 8251
TEST_SIZE = 3983
labels = ['programming',
          'style',
          'reference',
          'java',
          'web',
          'internet',
          'culture',
          'design',
          'education',
          'language',
          'books',
          'writing',
          'computer',
          'english',
          'politics',
          'history',
          'philosophy',
          'science',
          'religion',
          'grammar']

TRAIN_FILE = '../data/train-data.dat'
TEST_FILE = '../data/test-data.dat'
LABEL_TRAIN_FILE = '../data/train-label.dat'
LABEL_TEST_FILE = '../data/test-label.dat'

# create formatter
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():

    PartA()
    return 0


def PartA():
    logging.info("Reading data")
    x, y, test_x, test_y = ReadData()

    logging.info("Training model")
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
    clf.fit(x, y)
    logging.info("Predicting on test data")
    y_predicted = clf.predict(test_x)

    logging.info("Results:")
    acc = accuracy_score(test_y, y_predicted)
    logging.info('Accuracy {0}:'.format(str(acc)))
    logging.info('For each label:')
    index = 0
    for label in labels:
        acc = accuracy_score(test_y[:, index], y_predicted[:, index])
        logging.info('Accuracy {0}: {1}'.format(label, str(acc)))
        index += 1


def ReadData():
    # Read train data
    train_x = ReadValues(filename=TRAIN_FILE, file_size=TRAIN_SIZE)
    train_y = ReadLabels(filename=LABEL_TRAIN_FILE)

    # Read test data
    test_x = ReadValues(filename=TEST_FILE, file_size=TEST_SIZE)
    test_y = ReadLabels(filename=LABEL_TEST_FILE)

    # Return data
    return train_x, train_y, test_x, test_y


def ReadValues(filename, file_size):
    # Create new matrix filled with zero where rows are the number of documents
    # and columns are the number of words in the dictionary
    data = np.zeros((file_size, VOCAB_SIZE))
    # Initialize index
    index = 0
    with open(filename) as file:
        for line in file:
            # Split line into parts by whitespace
            parts = line.split()
            for part in parts:
                if part[0] != '<':
                    # Set 1 for this word in this document
                    data[index][int(part)] = 1
            index += 1
    return data


def ReadLabels(filename):
    data = pd.read_csv(filename, header=None, delim_whitespace=True, error_bad_lines=False).values
    return data


if __name__ == "__main__":
    main()
