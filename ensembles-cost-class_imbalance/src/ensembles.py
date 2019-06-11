from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import numpy as np
import math


class Ensembles:

    def __init__(self):
        self.bagging = None
        self.boosting = None
        self.randomForest = None
        self.voting = None
        self.ensembles = ["AdaBoost", "Bagging", "RandomForest", "Voting"]
        self.iterations = 10

        # Results
        self.accuracy_results = pd.DataFrame(
            columns=['Dataset', 'AdaBoost', 'Bagging', 'RandomForest', 'Voting'])

        # Data
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Prediction data
        self.y_predicted_train = None
        self.y_predicted_test = None

    def train(self, data_x, data_y, boosting=True, bagging=True, randomForest=True, voting=True, dataset=""):
        self.dataset = dataset
        self.setupResults()

        # Execute 10-fold cross validation and keep the mean score of train and test accuracy
        for i in range(0, self.iterations):
            # Create new data split
            self.createData(data_x, data_y)
            # Train and predict using Boosting ensemble
            if boosting is True:
                # Create model
                self.boosting = AdaBoostClassifier(random_state=0)
                # Train predict and log results for this model
                self.fitPredict(self.boosting, "AdaBoost")
            # Train and predict using RandomForest ensemble
            if randomForest is True:
                # Create model
                self.randomForest = RandomForestClassifier(
                    random_state=0, n_estimators=50)
                # Train predict and log results for this model
                self.fitPredict(self.randomForest, "RandomForest")
            # Train and predict using Bagging ensemble
            if bagging is True:
                # Create model
                self.bagging = BaggingClassifier(
                    random_state=0)
                # Train predict and log results for this model
                self.fitPredict(self.bagging, "Bagging")
            if voting is True:
                # Train and predict using Voting ensemble
                rf = RandomForestClassifier(random_state=0, n_estimators=10)
                knn = KNeighborsClassifier()
                svc = SVC(random_state=0, gamma='auto')
                mnb = MultinomialNB()

                self.voting = VotingClassifier(estimators=[(
                    'Random Forests', rf), ('KNeighbors', knn), ('SVC', svc), ('MultinomialNB', mnb)], voting='hard')
                # Train predict and log results for this model
                self.fitPredict(self.voting, "Voting")

        # Calculate mean for each ensemble
        self.meanResults()

        # Log final results
        for ensembe in self.ensembles:
            self.logEnsembleResults(ensembe)
        self.logResults()
        self.printEnsembleResults()

    def fitPredict(self, model, name):
        # Train model
        model.fit(self.x_train, self.y_train)
        # Predict
        self.y_predicted_train = model.predict(self.x_train)
        self.y_predicted_test = model.predict(self.x_test)
        # Log results
        self.addResults(name)

    def createData(self, data_x, data_y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data_x, data_y, test_size=0.30, random_state=0)

    def addResults(self, name):
        result_train = metrics.accuracy_score(self.y_train, self.y_predicted_train)
        result_test = metrics.accuracy_score(self.y_test, self.y_predicted_test)
        self.results[name]["train"] += result_train
        self.results[name]["test"] += result_test

    def meanResults(self):
        for ensemble in self.ensembles:
            self.results[ensemble]["train"] /= self.iterations
            self.results[ensemble]["test"] /= self.iterations
        self.calculateRanking()

    def calculateRanking(self):
        ensembles = self.ensembles.copy()
        i = 1
        while len(ensembles) > 0:
            max_ensemble = None
            duplicates = 0
            for ensemble in ensembles:
                if max_ensemble == None:
                    max_ensemble = ensemble
                else:
                    if self.results[max_ensemble]["test"] < self.results[ensemble]["test"]:
                        max_ensemble = ensemble
                        duplicates = 0
                    elif self.results[max_ensemble]["test"] == self.results[ensemble]["test"]:
                        duplicates += 1
            if duplicates == 0:
                self.ranking[max_ensemble] = " (" + str(i) + ")"
                ensembles.remove(max_ensemble)
            else:
                duplicates += 1
                ranking = i + (1 / duplicates)
                j = 0
                while duplicates > 0:
                    if self.results[max_ensemble]["test"] == self.results[ensembles[j]]["test"]:
                        self.ranking[ensembles[j]] = " (" + str(ranking) + ")"
                        ensembles.remove(ensembles[j])
                        duplicates -= 1
                    else:
                        j += 1
            i += 1

    def logResults(self):
        self.accuracy_results = self.accuracy_results.append({
            'Dataset': self.dataset,
            self.ensembles[0]: str(float("%0.4f" % self.results[self.ensembles[0]]["test"])) + self.ranking[self.ensembles[0]],
            self.ensembles[1]: str(float("%0.4f" % self.results[self.ensembles[1]]["test"])) + self.ranking[self.ensembles[1]],
            self.ensembles[2]: str(float("%0.4f" % self.results[self.ensembles[2]]["test"])) + self.ranking[self.ensembles[2]],
            self.ensembles[3]: str(float("%0.4f" % self.results[self.ensembles[3]]["test"])) + self.ranking[self.ensembles[3]]}, ignore_index=True)

    def logEnsembleResults(self, name):
        self.ensemble_results = self.ensemble_results.append({'Algorithm': name, 'Mean accuracy train':  float(
            "%0.3f" % self.results[name]["train"]), 'Mean accuracy test': float("%0.3f" % self.results[name]["test"])}, ignore_index=True)

    def printEnsembleResults(self):
        # Print final results
        print(self.ensemble_results)

    def printResults(self):
        # Print final results
        print(self.accuracy_results)
        print("")

    def setupResults(self):
        self.ensemble_results = pd.DataFrame(
            columns=['Algorithm', 'Mean accuracy train', 'Mean accuracy test'])
        self.results = {"Bagging": {"train": 0, "test": 0}, "AdaBoost": {
            "train": 0, "test": 0}, "Voting": {"train": 0, "test": 0}, "RandomForest": {"train": 0, "test": 0}}
        self.ranking = {}
