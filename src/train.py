#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:00:31 2020

@author: chenzhiyi
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

feature_file_name = "feature_more_balance_feature_unweighted.csv"
unique_value_limit = 20
n_instance = 299

parameter_space = {
    'RandomForestRegressor' : {
        'n_estimators' : [i for i in range(10, 30, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
        'min_samples_split' : [i for i in range(2, 20, 2)]
        }
    }

def readCSV(fname):
    index = [i for i in range(0, n_instance)]
    df = pd.read_csv(fname)
    df = df.sort_values(by=['instance'])
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
        print(temp)
        if temp in best_solver.iloc[i]:
            correct_count += 1
    return correct_count / len(scores)

def averageScore(predicted_scores, actual_scores, solvers) :
    result = 0
    for i in range(len(predicted_scores)) :
        predicted_solver = np.argmax(predicted_scores[i])
        result = result + actual_scores.iloc[i, predicted_solver]
    return result/len(predicted_scores)

def singleBestSolver(all_scores) :
    all_mean = all_scores.mean()
    return all_mean.max

def oracleAveScore(data):
    return data.mean()
        

def preprocessing(data, scaler) :
    data = scaler.transform(data)
    # data = pca.transform(data)
    return data


def expandFeature(feature) :
    col = feature.columns
    l = len(col)
    for i in range(1, l-1):
        for j in range(i, l):
            feature[col[i] + '*' + col[j]] = feature.iloc[:,i] * feature.iloc[:,j]
    return feature

    

feature = readCSV(feature_file_name)
feature = feature.fillna(0)
column_selection = (feature.nunique() > unique_value_limit)
feature = feature.loc[:,  column_selection]

feature = expandFeature(feature)

all_scores = readCSV("result_unweighted.csv")
best_solver = readCSV("per_instance_best_solver_unweighted.csv")
best_score = readCSV("per_instance_best_score_unweighted.csv")

rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)

for tr, te in rs.split(feature):
    train_ind = tr
    test_ind = te

solvers = list(all_scores.columns)
solvers.remove('instance')

all_reg = {}
all_pca = {}
inputInstance = feature.iloc[train_ind, 1:]
scaler = StandardScaler()
inputInstance = scaler.fit_transform(inputInstance)
all_scaler = scaler
for s in solvers:
    target = all_scores.loc[train_ind, s]
    
    # pca = PCA(n_components=6)
    # inputInstance = pca.fit_transform(inputInstance)
    # all_pca[s] = pca
    
    reg = RandomForestRegressor()
    clf = RandomizedSearchCV(reg, parameter_space['RandomForestRegressor'], random_state=0)
    search = clf.fit(inputInstance, target)
    print(search.best_params_)
    all_reg[s] = clf


test_instance = feature.iloc[test_ind, 1:]
test_best_solver = best_solver.loc[test_ind, 'best-solver(s)']
test_all_scores = all_scores.iloc[test_ind, 1:]
test_result = np.array([[] for i in range(len(test_instance))])
for s in solvers:
    processed_test_instance = preprocessing(test_instance, all_scaler)
    cur_result = all_reg[s].predict(processed_test_instance)
    cur_result = cur_result.reshape(len(cur_result), 1)
    test_result = np.hstack((test_result, cur_result))

print("Percentage of correct best solver predicted: ", accuracy(test_result, test_best_solver, solvers))

print("Average score of the prediction: ", averageScore(test_result, test_all_scores, solvers))

print(singleBestSolver(test_all_scores))

print(oracleAveScore(best_score.iloc[test_ind, 1:]))