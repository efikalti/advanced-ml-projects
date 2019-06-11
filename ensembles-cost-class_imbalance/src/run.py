#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sklearn.datasets as datasets
from sklearn import preprocessing
import csv
import numpy as np
import pandas as pd


# Assignment code
import ensembles
import cost
import class_imbalance


def main():

    partA()
    # partB()
    # partC()

    return 0


def partA():
    """ Assignment Part A """

    # Create ensemble object
    ensemble = ensembles.Ensembles()

    print("1.Iris dataset")
    runEnsembleForDataset(ensemble, dataset=datasets.load_iris(), name="iris")
    print("")

    print("2.Wine dataset")
    runEnsembleForDataset(ensemble, dataset=datasets.load_wine(), name="wine")
    print("")

    print("3.Digits dataset")
    runEnsembleForDataset(ensemble, dataset=datasets.load_digits(), name="digits")
    print("")

    print("4.Breast cancer dataset")
    runEnsembleForDataset(ensemble, dataset=datasets.load_breast_cancer(), name="breast cancer")
    print("")

    print("5.Abalone dataset")
    # read data/abalone.data
    data = pd.read_csv("data/abalone.data", sep=",")
    data_x = data.values[:, :-1]
    data_y = data.values[:, -1].astype('int')
    for i in range(0, len(data_x)):
        if data_x[i, 0] == 'M':
            data_x[i, 0] = 0
        elif data_x[i, 0] == 'F':
            data_x[i, 0] = 1
        else:
            data_x[i, 0] = 2
    runEnsembleForDataset(ensemble, data_x=data_x, data_y=data_y, name="abalone")
    print("")

    print("6.Heart dataset")
    # read data/heart.csv
    data = pd.read_csv("data/heart.csv", sep=",")
    data_x = data.values[:, :-1]
    data_y = data.values[:, -1]
    runEnsembleForDataset(ensemble, data_x=data_x, data_y=data_y, name="heart")
    print("")

    print("7.Glass dataset")
    # read data/glass.data
    data = pd.read_csv("data/glass.data", sep=",")
    data_x = data.values[:, :-1]
    data_y = data.values[:, -1]
    runEnsembleForDataset(ensemble, data_x=data_x, data_y=data_y, name="glass")
    print("")

    print("8.Transfusion dataset")
    # read data/transfusion.data
    data = pd.read_csv("data/transfusion.data", sep=",")
    data_x = data.values[:, :-1]
    data_y = data.values[:, -1]
    runEnsembleForDataset(ensemble, data_x=data_x, data_y=data_y, name="transfusion")
    print("")

    print("9.Starcraft dataset")
    # read data/SkillCraft1_Dataset.csv
    data = pd.read_csv("data/SkillCraft1_Dataset.csv", sep=",", na_values=['?'])
    data_x = data.iloc[:, 2:]
    data_x = data_x.fillna(data_x.mean())
    data_x = data_x.values
    data_y = data.values[:, 1]
    runEnsembleForDataset(ensemble, data_x=data_x, data_y=data_y, name="starcraft")
    print("")

    print("10.Credit Card dataset")
    # read data/creditcard.csv
    data = pd.read_csv("data/creditcard.csv", sep=",")
    data_x = data.values[:, :-1]
    data_y = data.values[:, -1]
    #runEnsembleForDataset(ensemble, data_x=data_x, data_y=data_y, name="credit card")
    print("")

    ensemble.printResults()


def runEnsembleForDataset(ensemble, dataset=None, name="", data_x=None, data_y=None):
    if dataset is not None:
        data_x = dataset['data']
        data_y = dataset['target']
    data_x = scaleData(data_x)
    ensemble.train(data_x, data_y, dataset=name)


def partB():
    """ Assignment Part B """

    # read data/heart.csv
    data = pd.read_csv("data/heart.csv", sep=",", dtype='float')
    data_x = data.values[:, :-1]
    data_y = data.values[:, -1].astype(int)

    data_x = scaleData(data_x)

    cost_matrix = [[0, 1], [5, 0]]
    costClassifier = cost.CostClassification(data_x, data_y, cost_matrix)
    costClassifier.run(withClassWeight=True, withStratification=True, withRejectionSampling=True)

    print(costClassifier.results)


def partC():
    """ Assignment Part C """

    # read data/creditcard.csv
    data = pd.read_csv("data/creditcard.csv", sep=",", dtype='float')
    data_x = data.values[:10000, :-1]
    data_y = data.values[:10000, -1].astype(int)

    data_x = scaleData(data_x)

    classImbalance = class_imbalance.ClassImbalance(data_x, data_y)
    classImbalance.setMinorityMajorityClass(minority=1, majority=0)
    classImbalance.run(withCostSensitive=True, withEasyEnsemble=True)
    print(classImbalance.results)


def scaleData(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_x = min_max_scaler.fit_transform(x)
    return scaled_x


if __name__ == "__main__":
    main()
