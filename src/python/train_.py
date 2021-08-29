#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:00:31 2020

@author: chenzhiyi
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from itertools import chain, combinations
import os

os.chdir("../../data")

feature_file_name = "feature_unweighted_shuffled.csv"
result_file_name = "result_unweighted_shuffled.csv"
best_score_file_name = "per_instance_best_score_unweighted_shuffled.csv"
# problem_label_file_name = "problem_label.csv"
n_instance = 297
n_iteration = 5
expand_n = 1

parameter_space = {
    'RandomForestRegressor' : {
        'n_estimators' : [i for i in range(10, 30, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
        'min_samples_split' : [i for i in range(2, 20, 2)]
        },
    'DecisionTreeRegressor' : {
        'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
        'splitter': ['best, random'],
        'max_depth': [2,3,4,5,8,10,15],
        'min_samples_split': [i for i in range(2, 20, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
    }
}

parameter_space_grid = {
    'RandomForestRegressor' : {
        'n_estimators' : [i for i in range(10, 60, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
        'min_samples_split' : [i for i in range(2, 20, 2)]
        },
    'DecisionTreeRegressor': {
        'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
        'max_depth': [2, 3, 4, 5, 8, 10, 15],
        'min_samples_split': [i for i in range(2, 20, 2)],
        'min_samples_leaf': [i for i in range(1, 10, 2)],
    }
}

best_param = {
    'Loandra' : {'criterion': 'mse', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4},
    'SATLike': {'criterion': 'mse', 'max_depth': 4, 'min_samples_leaf': 7, 'min_samples_split': 16},
    'LinSBPS2018': {'criterion': 'friedman_mse', 'max_depth': 5, 'min_samples_leaf': 7, 'min_samples_split': 18},
    'sls-mcs': {'criterion': 'mse', 'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 6},
    'sls-mcs-lsu': {'criterion': 'friedman_mse', 'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 6},
    'Open-WBO-g': {'criterion': 'mae', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 10},
    'Open-WBO-ms': {'criterion': 'friedman_mse', 'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 12},
}

def readCSV(fname):
    index = [i for i in range(0, n_instance)]
    df = pd.read_csv(fname)
    df.insert(0, 'ind', index)
    df = df.set_index('ind')
    return df

def accuracy(scores, best_solver, solvers) :
    predict_solvers = []
    correct_count = 0
    for i in range(len(scores)) :
        predict_solvers.append(solvers[np.argmax(scores[i])])
    for i in range(len(best_solver)) :
        temp = '\'' + predict_solvers[i] + '\''
        if temp in best_solver.iloc[i]:
            correct_count += 1
    return correct_count / len(scores)

def averageScore(predicted_scores, actual_scores) :
    result = 0
    for i in range(len(predicted_scores)) :
        predicted_solver = np.argmax(predicted_scores[i])
        result = result + actual_scores.iloc[i, predicted_solver]
    return result/len(predicted_scores)

def singleBestSolver(all_scores) :
    all_mean = all_scores.mean()
    return all_mean.max()

def oracleAveScore(data):
    return data.mean()
        

def preprocessing(data, scaler) :
    data = scaler.transform(data)
    # data = pca.transform(data)
    return data


def expandFeature(feature, n):
    new_feature = feature.copy()
    col = new_feature.columns
    l = len(col)
    indexs = np.arange(l)
    for i in range(2, n + 1):
        for j in range(l):
            new_feature[col[j] + "**" + str(i)] = np.power(new_feature.iloc[:,j], i)
        for j in combinations(indexs, i):
            names = '*'.join(np.array(col)[list(j)])
            new_col = np.ones(len(new_feature))
            for k in j:
                new_col = new_col*new_feature.iloc[:,k]
            new_feature[names] = new_col
    return new_feature

def resultAnalysis(predicted_scores, best_scores, true_scores):
    predict_error_per_instance = dict()
    for i in range(len(predicted_scores)) :
        predict_solver = np.argmax(predicted_scores[i])
        error =  best_score.iloc[i, 1] - true_scores.iloc[i, predict_solver]
        predict_error_per_instance[best_score.iloc[i, 0]] =  error
    return predict_error_per_instance


feature = readCSV(feature_file_name)
all_scores = readCSV(result_file_name)
feature = feature.iloc[:, 1:]
best_score = readCSV(best_score_file_name)

test_size = int(n_instance / n_iteration)
for i in range(n_iteration):
    start = test_size * i
    end = test_size * (i + 1)
    train_ind = [j for j in range(0, start)] + [j for j in range(end, n_instance)]
    test_ind = [j for j in range(start, end)]

    solvers = list(all_scores.columns)
    solvers.remove('instance')

    all_reg = {}
    all_param = {}
    inputInstance = feature.iloc[train_ind, :]
    scaler = StandardScaler()
    all_scaler = scaler


    for s in solvers:
        target = all_scores.loc[train_ind, s]


        reg = DecisionTreeRegressor(**best_param[s])
        #clf = GridSearchCV(reg, parameter_space_grid['DecisionTreeRegressor'], refit=True)
        #clf.fit(inputInstance, target)
        #all_param[s] = clf.best_params_
        #print(clf.best_params_)
        reg.fit(inputInstance, target)
        #print("{}: {}".format(s, reg.score(inputInstance, target)))
        all_reg[s] = reg


    test_instance = feature.iloc[test_ind, :]
    test_all_scores = all_scores.iloc[test_ind, 1:]
    test_result = np.array([[] for i in range(len(test_instance))])
    for s in solvers:
        processed_test_instance = pd.DataFrame(test_instance, columns=test_instance.columns)
        cur_result = all_reg[s].predict(processed_test_instance)
        cur_result = cur_result.reshape(len(cur_result), 1)
        test_result = np.hstack((test_result, cur_result))

    print("Average score of the prediction: ", averageScore(test_result, test_all_scores))

    print("Single Best Solver: ", singleBestSolver(test_all_scores))

    print(oracleAveScore(best_score.iloc[test_ind, 1:]))
print(end)
# print("Error per instance: ", resultAnalysis(test_result, best_score.iloc[test_ind], test_all_scores))










