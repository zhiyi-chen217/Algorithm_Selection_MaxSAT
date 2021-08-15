#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:00:31 2020

@author: chenzhiyi
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
import os


os.chdir("../../data")

feature_file_name = "feature_unweighted_shuffled.csv"
result_file_name = "result_unweighted_shuffled.csv"
best_score_file_name = "per_instance_best_score_unweighted_shuffled.csv"
unique_value_limit = 50
n_instance = 297
sum_rf = 0
sum_prob = 0
sum_prob_distribution = 0
sum_sb = 0
sum_oracle = 0
n_iteration = 5
special_instance = {}
parameter_space = {
    'RandomForestRegressor' : {
        'n_estimators' : [i for i in range(10, 30, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
        'min_samples_split' : [i for i in range(2, 20, 2)]
        }
    }

parameter_space_grid = {
    'RandomForestRegressor' : {
        'n_estimators' : [i for i in range(10, 60, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
        'min_samples_split' : [i for i in range(2, 20, 2)]
        }
    }

def calculateImportance(acc_importance, cur_importance, features):
    f_index = np.argsort(-cur_importance)[:10]
    features = np.array(features)
    
    for feature in features[f_index]:
        acc_importance[feature] += 1

def readCSV(fname):
    index = [i for i in range(0, n_instance)]
    df = pd.read_csv(fname)
    df = df.sort_values(by=['instance'])
    df.insert(0, 'ind', index)
    df = df.set_index('ind')
    return df

def accuracy(best_solver, solvers) :
    correct_count = 0
    for i in range(len(best_solver)) :
        temp = '\'' + solvers[i] + '\''
        #print(temp)
        if temp in best_solver.iloc[i]:
            correct_count += 1
    return correct_count / len(solvers)

def averageScore(predicted_scores, actual_scores) :
    result = 0
    for i in range(len(predicted_scores)) :
        predicted_solver = np.argmax(predicted_scores[i])
        result = result + actual_scores.iloc[i, predicted_solver]
    return result/len(predicted_scores)

def singleBestSolver(all_scores) :
    all_mean = all_scores.mean()
    return max(all_mean)

def oracleAveScore(data):
    return data.mean()
        

def preprocessing(data, scaler) :
    data = scaler.transform(data)
    # data = pca.transform(data)
    return data


def expandFeature(feature) :
    col = feature.columns
    l = len(col)
    for k in range(1, int(l-1 / 2)):
        i = np.random.randint(1, l)
        j = np.random.randint(1, l)
        feature[col[i] + '*' + col[j]] = feature.iloc[:,i] * feature.iloc[:,j]
    return feature

def resultAnalysis(predicted_scores, best_score, true_scores):
    d = pd.DataFrame(data=[], columns=["instance", "error"])
    for i in range(len(predicted_scores)) :
        predict_solver = np.argmax(predicted_scores[i])
        error =  best_score.iloc[i, 1] - true_scores.iloc[i, predict_solver]
        d = d.append({"instance" : best_score.iloc[i, 0], "error" : error}, ignore_index=True)
    return d
def resultAnalysisCluster(best_score, actual_scores):
    d = pd.DataFrame(data=[], columns=["instance", "error"])
    for i in range(len(actual_scores)):
        error =  best_score.iloc[i, 1] - actual_scores[i]
        d = d.append({"instance" : best_score.iloc[i, 0], "error" : error}, ignore_index=True)
    return d
def bestNSolver(centroid, instance, n):
    distance = {}
    for s in solvers:
        distance[s] = np.linalg.norm(centroid[s] -  instance)
    return list(dict(sorted(distance.items(), key=lambda item: item[1])).keys())[:n]

feature = readCSV(feature_file_name)
feature = feature.fillna(0)
column_selection = (feature.nunique() > unique_value_limit)
feature = feature.loc[:,  column_selection]
dim_reduction = feature.columns[1:]
#feature = expandFeature(feature)

all_scores = readCSV(result_file_name)
best_score = readCSV(best_score_file_name)

test_size = int(n_instance / n_iteration)
for i in range(n_iteration):
    start = test_size * i
    end = test_size * (i + 1)
    train_ind = [j for j in range(0, start)] + [j for j in range(end, n_instance)]
    test_ind = [j for j in range(start, end)]
    
    solvers = list(all_scores.columns)
    solvers.remove('instance')
    # acc_importance = {}
    # features = feature.columns[1:]
    # for f in features:
    #     acc_importance[f] = 0
        
    all_reg = {}
    inputInstance = feature.iloc[train_ind, 1:]
    scaler = StandardScaler()
    inputInstance = scaler.fit_transform(inputInstance)
    all_scaler = scaler
    
    # supervised learning
    for s in solvers:
        target = all_scores.loc[train_ind, s]
        
        reg = RandomForestRegressor()
        # reg = GridSearchCV(reg, parameter_space_grid['RandomForestRegressor'], cv=skf.split(inputInstance, problem_train_label))
        reg.fit(inputInstance, target)
        # print(reg.best_params_)
        # print(features[np.argmax(reg.feature_importances_)])
        #calculateImportance(acc_importance, reg.feature_importances_, features)
        all_reg[s] = reg
    

    # predict using model
    test_instance = feature.iloc[test_ind, 1:]
    test_all_scores = all_scores.iloc[test_ind, 1:]
    test_result = np.array([[] for i in range(len(test_instance))])
    for s in solvers:
        processed_test_instance = preprocessing(test_instance, all_scaler)
        cur_result = all_reg[s].predict(processed_test_instance)
        cur_result = cur_result.reshape(len(cur_result), 1)
        test_result = np.hstack((test_result, cur_result))

    print("Average score of the prediction: ", averageScore(test_result, test_all_scores))

    print("Single Best Solver: ", singleBestSolver(test_all_scores))

    print(oracleAveScore(best_score.iloc[test_ind, 1:]))









