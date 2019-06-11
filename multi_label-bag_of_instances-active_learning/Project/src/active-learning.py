#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import logging

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

VOCAB_SIZE = 8520
MOST_FREQ = 2

TRAIN_FILE = '../data/train-data.dat'
TEST_FILE = '../data/test-data.dat'
LABEL_TRAIN_FILE = '../data/train-label.dat'
LABEL_TEST_FILE = '../data/test-label.dat'

PROB_THRESHOLD = 0.6
SAMPLE_NUMBER = 50

# create formatter
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    PartC()
    return 0


def PartC():
    # Results dataframe
    results = pd.DataFrame(
        columns=['Accuracy Before Sampling', 'Accuracy After Sampling'])

    logging.info("Reading data")
    # Read train data
    train_x, train_y = ReadValues(x_filename=TRAIN_FILE, y_filename=LABEL_TRAIN_FILE)

    # Split training set to labeled and unlabeled, un_y will not be used
    labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(
        train_x, train_y, test_size=0.5, random_state=0)

    # Read test data
    test_x, test_y = ReadValues(x_filename=TEST_FILE, y_filename=LABEL_TEST_FILE)

    x = np.copy(labeled_x)
    y = np.copy(labeled_y)
    for i in range(0, 10):
        # Copy original dataset to new arrays
        logging.info("Train Naive Bayes model on labeled data")
        model = MultinomialNB()
        model.fit(x, y)
        logging.info("Predict on test data")
        y_predicted_before = model.predict(test_x)

        logging.info("Sample from unlabeled data using the probabilities predicted by the trained model")
        probabilities = model.predict_proba(unlabeled_x)

        samples_added = 0
        threshold = PROB_THRESHOLD
        samples_x = []
        samples_y = []
        while samples_added < SAMPLE_NUMBER:
            deletions = []
            for j in range(0, len(probabilities)):
                # Get the bigger of the probabilities
                max_prob = max(probabilities[j][0], probabilities[j][1])
                # If the max probability is smaller or equal to the threshold
                # Add that unlabeled data to the labeled dataset
                if max_prob <= threshold and samples_added < SAMPLE_NUMBER:
                    samples_x.append(unlabeled_x[j])
                    samples_y.append(unlabeled_y[j])
                    deletions.append(j)
                    samples_added += 1
            if samples_added < SAMPLE_NUMBER:
                # Increase threshold by 10%
                threshold += 0.1
            # Delete probabilities added
            probabilities = np.delete(probabilities, deletions, axis=0)
            if len(probabilities) < SAMPLE_NUMBER - samples_added:
                samples_added = (SAMPLE_NUMBER - samples_added) - len(probabilities)
                threshold = 1
        # Add new labeled data
        x = np.concatenate((x, samples_x), axis=0)
        y = np.concatenate((y, samples_y), axis=0)
        # Remove from unlabeled dataset
        unlabeled_x = np.delete(unlabeled_x, deletions, axis=0)
        unlabeled_y = np.delete(unlabeled_y, deletions, axis=0)

        logging.info("Train Naive Bayes model on the new labeled data")
        model = MultinomialNB()
        model.fit(x, y)
        logging.info("Predict again on test data")
        y_predicted_after = model.predict(test_x)

        acc_b = accuracy_score(test_y, y_predicted_before)
        logging.info(
            'Accuracy on test data before adding from the unlabeled data {0}:'.format(str(acc_b)))
        acc_a = accuracy_score(test_y, y_predicted_after)
        logging.info(
            'Accuracy on test data after adding from the unlabeled data {0}:'.format(str(acc_a)))

        results = results.append({'Accuracy Before Sampling': float("%0.5f" % acc_b),
                                  'Accuracy After Sampling': float("%0.5f" % acc_a)}, ignore_index=True)

    logging.info('Final results:')
    logging.info('\n' + str(results))


def ReadValues(x_filename, y_filename):

    # Read labels array
    y_data = pd.read_csv(y_filename, header=None, delim_whitespace=True,
                         error_bad_lines=False).values

    # Create new matrix filled with zero where rows are the number of documents
    # and columns are the number of words in the dictionary
    x_data = np.zeros((y_data.shape[0], VOCAB_SIZE))
    y_data_transformed = np.zeros((y_data.shape[0]))

    # Initialize index
    index = 0
    with open(x_filename) as file:
        for line in file:
            # Number of sentences in this document
            if y_data[index][MOST_FREQ] == 1:
                y_data_transformed[index] = 1

            # Split line into parts by whitespace
            parts = line.split()
            for part in parts:
                if part[0] != '<':
                    # Set 1 for this word in this document
                    x_data[index][int(part)] = 1
            index += 1
    return x_data, y_data_transformed


def max(a, b):
    if a >= b:
        return a
    return b


if __name__ == "__main__":
    main()
