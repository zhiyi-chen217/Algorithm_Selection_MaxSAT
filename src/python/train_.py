#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:00:31 2020

@author: chenzhiyi
"""
import numpy as np
import pandas as pd
import feature_selection
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from itertools import chain, combinations
import os

os.chdir("../data")

feature_file_name = "feature_extended_unweighted.csv"
result_file_name = "result_unweighted.csv"
best_solver_file_name = "per_instance_best_solver_unweighted.csv"
best_score_file_name = "per_instance_best_score_unweighted.csv"
# problem_label_file_name = "problem_label.csv"
n_instance = 297
expand_n = 1

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


instance_features = ['instance', 'ncls', 'nhard_len_stats.ave',
       'nhard_len_stats.max', 'nhard_len_stats.min', 'nhard_len_stats.stddev',
       'nhards', 'nsoft_len_stats.ave', 'nsoft_len_stats.max',
       'nsoft_len_stats.min', 'nsoft_len_stats.stddev', 'nsoft_wts', 'nsofts',
       'nvars', 'soft_wt_stats.ave', 'soft_wt_stats.max', 'soft_wt_stats.min',
       'soft_wt_stats.stddev']
balance_features = ['balance-hard', 'balance-hard-max', 'balance-hard-mean',
       'balance-hard-min', 'balance-hard-std', 'balance-soft',
       'balance-soft-max', 'balance-soft-mean', 'balance-soft-min',
       'balance-soft-std']
graph_features = ['VG-mean', 'VG-max', 'VG-min', 'VG-std', 'VG-entropy', 'VCG_C-entropy', 'VCG_C-max', 'VCG_C-mean',
                  'VCG_C-min', 'VCG_C-std', 'VCG_V-entropy', 'VCG_V-max', 'VCG_V-mean', 'VCG_V-min', 'VCG_V-std']
horn_features = ['Horn-fraction', 'Horn-V-mean', 'Horn-V-max', 'Horn-V-min', 'Horn-V-std']

used_features = instance_features + graph_features
feature = readCSV(feature_file_name)
#feature = feature.loc[:,  used_features]
all_scores = readCSV(result_file_name)
feature = feature.fillna(0)
# all_solver_features = feature_selection.selectFeature(feature, all_scores, expand_n)
feature = feature.iloc[:, 1:]
feature = expandFeature(feature, expand_n)

best_solver = readCSV(best_solver_file_name)
best_score = readCSV(best_score_file_name)
#problem_label = readCSV(problem_label_file_name)

rs = ShuffleSplit(n_splits=1, test_size=.25)

for tr, te in rs.split(feature):
    train_ind = tr
    test_ind = te

solvers = list(all_scores.columns)
solvers.remove('instance')

all_reg = {}
inputInstance = feature.iloc[train_ind, :]
scaler = StandardScaler()
inputInstance = pd.DataFrame(scaler.fit_transform(inputInstance), columns=inputInstance.columns)
all_scaler = scaler

# problem_train_label = list(problem_label.iloc[train_ind, 1])
# problem_test_label = list(problem_label.iloc[test_ind, 1])
skf = StratifiedKFold(n_splits=3)

for s in solvers:
    target = all_scores.loc[train_ind, s]
    
    # pca = PCA(n_components=6)
    # inputInstance = pca.fit_transform(inputInstance)
    # all_pca[s] = pca
    
    reg = DecisionTreeRegressor()
    # clf = GridSearchCV(reg, parameter_space_grid['RandomForestRegressor'], cv=skf.split(inputInstance, problem_train_label))
    # clf.fit(inputInstance.loc[:, all_solver_features[s]], target)
    reg.fit(inputInstance, target)
    r = export_text(reg, feature_names=list(inputInstance.columns))
    print("for {}:".format(s))
    print(r)
    # print("{}: {}".format(s, reg.score(inputInstance.loc[:, all_solver_features[s]], target)))
    all_reg[s] = reg


test_instance = feature.iloc[test_ind, :]
test_best_solver = best_solver.iloc[test_ind]['best-solver(s)']
test_all_scores = all_scores.iloc[test_ind, 1:]
test_result = np.array([[] for i in range(len(test_instance))])
for s in solvers:
    processed_test_instance = pd.DataFrame(preprocessing(test_instance, all_scaler), columns=test_instance.columns)
    cur_result = all_reg[s].predict(processed_test_instance)
    cur_result = cur_result.reshape(len(cur_result), 1)
    test_result = np.hstack((test_result, cur_result))
print("Percentage of correct best solver predicted: ", accuracy(test_result, test_best_solver, solvers))

print("Average score of the prediction: ", averageScore(test_result, test_all_scores))

print("Single Best Solver: ", singleBestSolver(test_all_scores))

print(oracleAveScore(best_score.iloc[test_ind, 1:]))

# print("Error per instance: ", resultAnalysis(test_result, best_score.iloc[test_ind], test_all_scores))










