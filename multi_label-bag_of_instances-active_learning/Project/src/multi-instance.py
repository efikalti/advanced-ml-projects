#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import logging

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

VOCAB_SIZE = 8520

MOST_FREQ = 2
MOST_FREQ_COUNT = 3181

TRAIN_FILE = '../data/train-data.dat'
TEST_FILE = '../data/test-data.dat'
LABEL_TRAIN_FILE = '../data/train-label.dat'
LABEL_TEST_FILE = '../data/test-label.dat'

SAMPLE_COEF = 1000

# create formatter
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    PartB()
    return 0


def PartB():
    logging.info("Reading data")
    # Read train data
    train_x, train_y = ReadValues(x_filename=TRAIN_FILE, y_filename=LABEL_TRAIN_FILE)

    # Read test data
    test_x, test_y = ReadValues(x_filename=TEST_FILE, y_filename=LABEL_TEST_FILE)

    logging.info("Train SVM model")
    model = SVC(gamma='auto', class_weight="balanced")
    model.fit(train_x, train_y)

    logging.info("Predict test data")
    y_predicted = model.predict(test_x)
    acc = accuracy_score(test_y, y_predicted)
    logging.info('Accuracy {0}:'.format(str(acc)))

    print(CalculateResults(test_x, y_predicted))


def ReadValues(x_filename, y_filename):

    file_ids = FileIDs(x_filename)

    # Read labels array
    y_data = pd.read_csv(y_filename, header=None, delim_whitespace=True,
                         error_bad_lines=False).values

    # Create new array to hold the transformed data
    x_data = []
    y_data_transformed = []

    index = 0
    with open(x_filename) as file:
        for line in file:
            if index % SAMPLE_COEF == 0:

                # Find the file id for this line
                file = int(re.findall('^<([0-9]*)>', line)[0])

                sentence = np.zeros((VOCAB_SIZE + 1), dtype=np.int32)
                sentence[-1] = file

                # Find label
                if y_data[index][MOST_FREQ] == 1:
                    label = 1
                else:
                    label = 0

                # Split line into parts by whitespace
                parts = line.split()
                file_id = True
                for part in parts:
                    if part[0] != '<':
                        # Set 1 for this word in this sentence
                        sentence[int(part)] = 1
                    else:
                        if file_id is False:
                            x_data.append(sentence)
                            y_data_transformed.append(label)
                            sentence = np.zeros((VOCAB_SIZE + 1), dtype=np.int32)
                            sentence[-1] = file
                        else:
                            file_id = False
            # Increase index
            index += 1
    return x_data, y_data_transformed


def CalculateResults(data, results):
    file_results = {}
    index = 0
    for result in results:
        file_id = str(data[index][-1])
        if file_id in file_results:
            file_results[file_id][result] += 1
        else:
            file_results[file_id] = [0, 0]
        index += 1
    return file_results


def NumOfClasses(filename):
    data = pd.read_csv(filename, header=None, delim_whitespace=True, error_bad_lines=False).values
    counts = np.zeros((data.shape[1]))
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            if data[i][j] == 1:
                counts[j] += 1
    return counts


def FileIDs(filename):
    doc_ids = {}
    index = 0
    with open(filename) as file:
        for line in file:
            # Find the file id for this line
            file = re.findall('^<([0-9]*)>', line)[0]
            if file not in doc_ids:
                doc_ids[file] = index
                index += 1
    return doc_ids


if __name__ == "__main__":
    main()
