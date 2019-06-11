import math
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

from cost import CostClassification


class ClassImbalance:

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.createData()
        self.setupResults()
        self.model = None
        self.min_sum = 0
        self.maj_sum = 0
        self.iterations = 5

    def run(self, withCostSensitive=True, withEasyEnsemble=True):
        self.logDataInfo(self.data_y)

        if withCostSensitive is True:
            # Run cost sensitive learning
            self.runCostSensitiveLearning()

        if withEasyEnsemble is True:
            # Run Easy Ensemble
            self.runEasyEnsemble()

    def runCostSensitiveLearning(self):

        print("Running Cost Sensitive Learning ---------------")
        minority_cost = math.floor(self.maj_sum / self.min_sum)
        self.cost_matrix = [[0, 1], [minority_cost, 0]]
        print("Cost matrix for cost sensitive learning:")
        print(self.cost_matrix)

        costSensitive = CostClassification(
            self.data_x, self.data_y, self.cost_matrix, classA=0, classB=1)
        costSensitive.run(withStratification=True, withClassWeight=True,
                          withRejectionSampling=True)
        print(costSensitive.results)
        print("----------------------------------------------")

    def runEasyEnsemble(self):

        print("")
        print("Running Easy Ensemble -----------------------------")
        # Copy original sample set
        x_train = np.copy(self.x_train)
        y_train = np.copy(self.y_train)

        predictions = []

        # Separate minority from majority in different arrays
        self.separateClasses()

        for i in range(0, self.iterations):
            # Run EasyEnsemble to create new dataset
            self.easyEnsembleSample()

            # Train a boosting model on the new data created by the easyEnsemble
            boost_model = AdaBoostClassifier(random_state=0)
            boost_model.fit(self.x_train_new, self.y_train_new)
            y_predicted_test = boost_model.predict(self.x_test)
            predictions.append(y_predicted_test)

        # Log results
        self.votingResults(predictions)

        # Restore original data
        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)

    def separateClasses(self):
        self.x_train_min = []
        self.y_train_min = []
        self.x_train_maj = []
        self.y_train_maj = []
        for sample in range(0, len(self.x_train)):
            if self.y_train[sample] == self.minority:
                self.x_train_min.append(self.x_train[sample])
                self.y_train_min.append(self.y_train[sample])
            else:
                self.x_train_maj.append(self.x_train[sample])
                self.y_train_maj.append(self.y_train[sample])

    def easyEnsembleSample(self):
        # Set legth for majority class, equal to minority sum
        majority_sum = len(self.x_train_min)

        self.x_train_new = []
        self.y_train_new = []
        added = []
        while majority_sum > 0:
            # Select a random sample
            sample = random.randint(0, len(self.y_train_maj)-1)
            while sample in added:
                sample = random.randint(0, len(self.y_train_maj)-1)
            added.append(sample)
            # Add new sample
            self.x_train_new.append(self.x_train_maj[sample])
            self.y_train_new.append(self.y_train_maj[sample])
            majority_sum -= 1

        # Add minority samples
        self.x_train_new.extend(self.x_train_min)
        self.y_train_new.extend(self.y_train_min)

        # Append class values to x_train_new in order to shuffle them
        for i in range(0, len(self.x_train_new)):
            self.x_train_new[i] = np.append(self.x_train_new[i], self.y_train_new[i])
        # Shuffle x_train_new
        np.random.shuffle(self.x_train_new)
        # Separate data from class values
        for i in range(0, len(self.x_train_new)):
            self.y_train_new[i] = self.x_train_new[i][-1].astype(int)
            self.x_train_new[i] = np.delete(self.x_train_new[i], -1)

    def votingResults(self, predictions):
        results = []
        # Calculate results with hard voting
        for index in range(0, len(self.x_test)):
            minority_sum = 0
            majority_sum = 0
            for model in range(0, len(predictions)):
                if predictions[model][index] == self.minority:
                    minority_sum += 1
                else:
                    majority_sum += 1
            if minority_sum >= majority_sum:
                results.append(self.minority)
            else:
                results.append(self.majority)
        self.logResults(results)

    def logResults(self, y_predicted_test):
        result_test = metrics.accuracy_score(self.y_test, y_predicted_test)

        print("Test results: " + str(result_test) + " acc")
        confusion_matrix = metrics.confusion_matrix(self.y_test, y_predicted_test, labels=[0, 1])
        print(confusion_matrix)

        self.results = self.results.append({'Class Imbalance Method': 'EasyEnsemble', 'Accuracy test': float("%0.3f" % result_test),
                                            'True min': confusion_matrix[self.minority][self.minority],
                                            'False min': confusion_matrix[self.minority][self.majority],
                                            'True maj': confusion_matrix[self.majority][self.majority],
                                            'False maj': confusion_matrix[self.majority][self.minority]}, ignore_index=True)
        print("----------------------------------------------")

    def logDataInfo(self, y):
        self.min_sum = 0
        self.maj_sum = 0
        for sample in y:
            if sample == self.minority:
                self.min_sum += 1
            else:
                self.maj_sum += 1
        print("Data information -----------------------------")
        print("Sample size:" + str(len(y)))
        print("Majority class sample size:" + str(self.maj_sum))
        print("Minority class sample size:" + str(self.min_sum))
        print("----------------------------------------------")

    def createData(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data_x, self.data_y, test_size=0.30, random_state=0)

    def setupResults(self):
        self.results = pd.DataFrame(
            columns=['Class Imbalance Method', 'Accuracy test',
                     'True min', 'False min', 'True maj', 'False maj'])

    def setMinorityMajorityClass(self, minority, majority):
        self.minority = minority
        self.majority = majority
