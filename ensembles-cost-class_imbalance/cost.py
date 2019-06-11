from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import math
import random


class CostClassification:

    def __init__(self, data_x, data_y, cost_matrix, classA=1, classB=2):
        self.data_x = data_x
        self.data_y = data_y
        self.cost = cost_matrix
        self.classA = classA
        self.classB = classB
        self.createData()
        self.setupResults()
        self.model = None
        self.class_weight = {
            classA: self.cost[0][1], classB: self.cost[1][0]}

        # Set min and max keys
        self.min_key = None
        for key in self.class_weight:
            if self.min_key is None:
                self.min_key = key
                self.min_pos = 0
                self.max_pos = 1
            else:
                if self.class_weight[key] < self.class_weight[self.min_key]:
                    self.min_key = key
                    self.min_pos = 1
                    self.max_pos = 0
        for key in self.class_weight:
            if self.min_key != key:
                self.max_key = key
        self.createSampleWeights(self.y_train)

    def run(self, withClassWeight=True, withStratification=True, withRejectionSampling=True):

        self.logDataInfo(self.y_train)

        print("Run without using the cost matrix -----------------------------")
        self.runWithoutCost(method="Without Cost")
        print("")

        if withClassWeight is True:
            print("Run using class weights as cost -------------------------------")
            self.runForClassWeights(method="With Class Weights")
            print("")

        if withStratification is True:
            print("Run using Stratification - Combination ------------------------")
            self.runForStratification(method="With Stratification")
            print("")

        if withRejectionSampling is True:
            print("Run using Rejection Sampling ----------------------------------")
            self.runForRejectionSampling(method="With Rejection Sampling")
            print("")

    def runWithoutCost(self, **kwargs):

        print("Running Linear SVM without using the cost matrix")
        self.LinearSVM(with_class_weight=False, **kwargs)

        print("Running Random Forest without using the cost matrix")
        self.RandomForest(with_class_weight=False, **kwargs)

        print("Running Naive Bayes without using the cost matrix")
        self.NaiveBayes(with_class_weight=False, **kwargs)

    def runForClassWeights(self, **kwargs):

        print("Running Linear SVM using the cost matrix as class weights")
        self.LinearSVM(with_class_weight=True, **kwargs)

        print("Running Random Forest using the cost matrix as class weights")
        self.RandomForest(with_class_weight=True, **kwargs)

        print("Running Naive Bayes using the cost matrix as class weights")
        self.NaiveBayes(with_class_weight=True, **kwargs)

    def runForStratification(self, **kwargs):
        # Copy original sample set
        x_train = np.copy(self.x_train)
        y_train = np.copy(self.y_train)
        # Log original dataset info
        self.logDataInfo(self.y_train)
        # Create new dataset
        size = len(x_train) * 2
        self.Stratification(size=size)
        self.x_train = self.x_train_new
        self.y_train = self.y_train_new
        # Log new dataset info
        self.logDataInfo(self.y_train_new)

        print("Running Linear SVM for Stratification")
        self.LinearSVM(with_class_weight=False, **kwargs)

        print("Running Random Forest for Stratification")
        self.RandomForest(with_class_weight=False, **kwargs)

        print("Running Naive Bayes for Stratification")
        self.NaiveBayes(with_class_weight=False, **kwargs)

        # Restore original data
        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)

    def runForRejectionSampling(self, **kwargs):
        # Copy original sample set
        x_train = np.copy(self.x_train)
        y_train = np.copy(self.y_train)
        # Log original dataset info
        self.logDataInfo(self.y_train)
        # Create new dataset
        self.RejectionSampling()
        self.x_train = self.x_train_new
        self.y_train = self.y_train_new
        # Log new dataset info
        self.logDataInfo(self.y_train_new)

        print("Running Linear SVM")
        self.LinearSVM(with_class_weight=False, **kwargs)

        print("Running Random Forest")
        self.RandomForest(with_class_weight=False, **kwargs)

        print("Running Naive Bayes")
        self.NaiveBayes(with_class_weight=False, **kwargs)

        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)

    def NaiveBayes(self, with_class_weight=True, **kwargs):
        # Create model using class weights according to the cost matrix
        self.model = MultinomialNB()
        self.train(with_class_weight=with_class_weight, algorithm='NaiveBayes', **kwargs)

    def RandomForest(self, with_class_weight=True, **kwargs):
        if with_class_weight is True:
            # Create model using class weights according to the cost matrix
            self.model = RandomForestClassifier(
                class_weight=self.class_weight, n_estimators=10, random_state=0)
        else:
            self.model = RandomForestClassifier(n_estimators=10, random_state=0)
        self.train(algorithm='RandomForest', **kwargs)

    def LinearSVM(self, with_class_weight=True, **kwargs):
        if with_class_weight is True:
            # Create model using class weights according to the cost matrix
            self.model = SVC(random_state=0, class_weight=self.class_weight, kernel='linear')
        else:
            # Create model without class weights
            self.model = SVC(random_state=0, kernel='linear')
        self.train(algorithm='LinearSVM', **kwargs)

    def train(self, with_class_weight=False, **kwargs):
        # Fit model using train data
        if with_class_weight is False:
            self.model.fit(self.x_train, self.y_train)
        else:
            self.model.fit(self.x_train, self.y_train, sample_weight=self.weights)
        # Predict on train
        self.y_predicted_train = self.model.predict(self.x_train)
        # Predict on test
        self.y_predicted_test = self.model.predict(self.x_test)
        # Log results
        self.logResults(**kwargs)

    # Modify the data so the frequency of each class is analogous to its misclassification cost
    def Stratification(self, size=None):
        if size is None:
            # Set the size the same as the current sample
            size = len(self.x_train)
        if size <= 0:
            # Set the size the same as the current sample
            size = len(self.x_train)

        # Calculate the ratio between the classes samples
        ratio = self.class_weight[self.max_key] / self.class_weight[self.min_key]

        # Calculate the number of samples for each class
        cost_sum = ratio + 1
        max_cost_size = math.floor(size / cost_sum) * ratio + (size % cost_sum)
        min_cost_size = math.floor(size / cost_sum)

        self.x_train_new = []
        self.y_train_new = []
        while max_cost_size > 0 or min_cost_size > 0:
            # Select a random sample
            sample = random.randint(0, len(self.y_train)-1)
            if self.y_train[sample] == self.min_key:
                if min_cost_size > 0:
                    self.x_train_new.append(self.x_train[sample])
                    self.y_train_new.append(self.y_train[sample])
                    min_cost_size -= 1
            else:
                if max_cost_size > 0:
                    self.x_train_new.append(self.x_train[sample])
                    self.y_train_new.append(self.y_train[sample])
                    max_cost_size -= 1

    # Rejection sampling with z = max cost
    def RejectionSampling(self):
        size = len(self.x_train)

        # Set z
        z = self.class_weight[self.max_key]

        self.x_train_new = []
        self.y_train_new = []
        for index in range(0, size):
            value = self.class_weight[self.y_train[index]] / z
            prob = random.uniform(0, 1)
            if (value >= prob):
                self.x_train_new.append(self.x_train[index])
                self.y_train_new.append(self.y_train[index])

    def createSampleWeights(self, y):
        self.weights = np.array(y)
        index = 0
        for sample in y:
            self.weights[index] = self.class_weight[sample]
            index += 1

    def logResults(self, method, algorithm):
        result_train = metrics.accuracy_score(self.y_train, self.y_predicted_train)
        result_test = metrics.accuracy_score(self.y_test, self.y_predicted_test)
        print("Training results: " + str(result_train) + " acc")

        print("Test results: " + str(result_test) + " acc")
        confusion_matrix = metrics.confusion_matrix(
            self.y_test, self.y_predicted_test, labels=[self.classA, self.classB])
        print(confusion_matrix)
        self.results = self.results.append({'Cost Sensitive Method': method, 'Algorithm': algorithm, 'Accuracy train':  float(
            "%0.3f" % result_train), 'Accuracy test': float("%0.3f" % result_test),
            'True max cost': confusion_matrix[self.max_pos][self.max_pos], 'False max cost': confusion_matrix[self.max_pos][self.min_pos],
            'True min cost': confusion_matrix[self.min_pos][self.min_pos], 'False min cost': confusion_matrix[self.min_pos][self.max_pos],
            'Sample size': len(self.x_train), 'Max cost sample': self.max_sum, 'Min cost sample': self.min_sum}, ignore_index=True)

        print("---------------------------------------------------------------")

    def logDataInfo(self, y):
        self.min_sum = 0
        self.max_sum = 0
        for sample in y:
            if sample == self.min_key:
                self.min_sum += 1
            else:
                self.max_sum += 1
        print("Sample size:" + str(len(self.x_train)))
        print("Max cost sample size:" + str(self.max_sum))
        print("Min cost sample size:" + str(self.min_sum))
        print("")

    def createData(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data_x, self.data_y, test_size=0.30)

    def setupResults(self):
        self.results = pd.DataFrame(
            columns=['Cost Sensitive Method', 'Algorithm', 'Accuracy train', 'Accuracy test',
                     'True max cost', 'False max cost', 'True min cost', 'False min cost',
                     'Sample size', 'Max cost sample', 'Min cost sample'])
