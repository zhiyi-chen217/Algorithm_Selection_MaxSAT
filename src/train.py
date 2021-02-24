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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import export_text
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import graphviz 
import os
import ast


os.chdir("../data")

feature_file_name = "feature_balance_graph_horn_feature_unweighted.csv"
unique_value_limit = 50
n_instance = 299
n_cluster = 3

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

def resultAnalysis(predicted_scores, best_scores, true_scores):
    for i in range(len(predicted_scores)) :
        predict_solver = np.argmax(predicted_scores[i])
        error =  best_score.iloc[i, 1] - true_scores.iloc[i, predict_solver]
        print(best_score.iloc[i, 0] + ":" + str(error))

    
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
graph_features = ['VG-mean', 'VG-max', 'VG-min', 'VG-std',
       'VCG-mean', 'VCG-max', 'VCG-min', 'VCG-std']
horn_features = ['Horn-fraction',
       'Horn-V-mean', 'Horn-V-max', 'Horn-V-min', 'Horn-V-std']
all_features = instance_features + balance_features + graph_features + horn_features
used_features = all_features
feature = readCSV(feature_file_name)
feature = feature.loc[:,  used_features]

feature = feature.fillna(0)
column_selection = (feature.nunique() > unique_value_limit)
feature = feature.loc[:,  column_selection]

#feature = expandFeature(feature)

all_scores = readCSV("result_unweighted.csv")
best_solver = readCSV("per_instance_best_solver_unweighted.csv")
best_score = readCSV("per_instance_best_score_unweighted.csv")
problem_label = readCSV("problem_label.csv")

rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)

for tr, te in rs.split(feature):
    train_ind = tr
    test_ind = te

solvers = list(all_scores.columns)
solvers.remove('instance')
acc_importance = {}
features = feature.columns[1:]
for f in features:
    acc_importance[f] = 0
    
all_reg = {}
all_pca = {}
inputInstance = feature.iloc[train_ind, 1:]
scaler = StandardScaler()
inputInstance = scaler.fit_transform(inputInstance)
all_scaler = scaler

problem_train_label = list(problem_label.iloc[train_ind, 1])
problem_test_label = list(problem_label.iloc[test_ind, 1])
skf = StratifiedKFold(n_splits=3)

# supervised learning
for s in solvers:
    target = all_scores.loc[train_ind, s]
    
    reg = DecisionTreeRegressor()
    # reg = GridSearchCV(reg, parameter_space_grid['RandomForestRegressor'], cv=skf.split(inputInstance, problem_train_label))
    reg.fit(inputInstance, target)
    # print(reg.best_params_)
    # print(features[np.argmax(reg.feature_importances_)])
    calculateImportance(acc_importance, reg.feature_importances_, features)
    all_reg[s] = reg

# unsupervised clustering
dim_reduction = list(dict(sorted(acc_importance.items(), key=lambda item: -item[1])).keys())[:10]
input_cluster = feature.loc[train_ind, dim_reduction]
kmeans = KMeans(n_clusters=n_cluster, random_state=0, init='k-means++').fit(input_cluster)
clusters = kmeans.predict(input_cluster)
cluster_best_solver = {}
for i in range(n_cluster):
    cluster_best_solver[i] = {}
train_best_solver = best_solver.loc[train_ind, 'best-solver(s)']
for i in range(len(train_best_solver)):
    c = clusters[i]
    cur_best_solver = cluster_best_solver[c]
    for b in ast.literal_eval(train_best_solver.iloc[i]):
        if b in cur_best_solver:
            cur_best_solver[b] += 1
        else:
            cur_best_solver[b] = 1

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









