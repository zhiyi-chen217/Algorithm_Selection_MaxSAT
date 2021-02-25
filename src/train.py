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
from sklearn.cluster import KMeans, Birch
import graphviz 
import os
import ast


os.chdir("../data")

feature_file_name = "feature_extended_unweighted.csv"
unique_value_limit = 50
n_instance = 297
n_cluster = 4
sum_cluster = 0
sum_rf = 0

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
dim_reduction = ['balance-hard-std',
    'ncls',
    'nsofts',
    'nvars',
    'VG-mean',
    'VG-max']
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

def averageScore(actual_scores, solvers) :
    result = 0
    for i in range(len(solvers)) :
        result = result + actual_scores.iloc[i][solvers[i]]
    return result/len(solvers)

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

    
instance_features = ['instance', 'ncls', 'nhard_len_stats.ave',
       'nhard_len_stats.max', 'nhard_len_stats.min', 'nhard_len_stats.stddev',
       'nhards', 'nsoft_len_stats.ave', 'nsoft_len_stats.max',
       'nsoft_len_stats.min', 'nsoft_len_stats.stddev', 'nsoft_wts', 'nsofts',
       'nvars', 'soft_wt_stats.ave', 'soft_wt_stats.max', 'soft_wt_stats.min',
       'soft_wt_stats.stddev']
balance_features = ['balance-hard', 'balance-hard-max', 'balance-hard-mean',
       'balance-hard-min', 'balance-hard-std', 'balance-hard-entropy','balance-soft',
       'balance-soft-max', 'balance-soft-mean', 'balance-soft-min',
       'balance-soft-std', 'balance-soft-entropy']
graph_features = ['VG-mean',
       'VG-max', 'VG-min', 'VG-std', 'VG-entropy', 'VCG_C-entropy',
       'VCG_C-max', 'VCG_C-mean', 'VCG_C-min', 'VCG_C-std', 'VCG_V-entropy',
       'VCG_V-max', 'VCG_V-mean', 'VCG_V-min', 'VCG_V-std']
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

for i in range(0, 40, 3):
    rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=i)
    
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
        
        reg = RandomForestRegressor()
        # reg = GridSearchCV(reg, parameter_space_grid['RandomForestRegressor'], cv=skf.split(inputInstance, problem_train_label))
        reg.fit(inputInstance, target)
        # print(reg.best_params_)
        # print(features[np.argmax(reg.feature_importances_)])
        #calculateImportance(acc_importance, reg.feature_importances_, features)
        all_reg[s] = reg
    
    # unsupervised clustering
    scaler_cluster = StandardScaler()
    input_cluster = scaler_cluster.fit_transform(feature.loc[train_ind, dim_reduction])
    cluster_best_solver = {}
    for s in solvers:
        cluster_best_solver[s] = []
    train_best_solver = best_solver.loc[train_ind, 'best-solver(s)']
    for i in range(len(train_best_solver)):
        instance = input_cluster[i, :]
        for b in ast.literal_eval(train_best_solver.iloc[i]):
            if len(cluster_best_solver[b]) == 0:
                cluster_best_solver[b] = np.array([instance])
            else:
                cluster_best_solver[b] = np.vstack((cluster_best_solver[b], instance))
    centroid = {}
    for s in solvers:
        centroid[s] = np.mean(cluster_best_solver[s], axis = 0)
    
    test_instance = feature.iloc[test_ind, 1:]
    test_best_solver = best_solver.loc[test_ind, 'best-solver(s)']
    test_all_scores = all_scores.iloc[test_ind, 1:]
    test_result = np.array([[] for i in range(len(test_instance))])
    for s in solvers:
        processed_test_instance = preprocessing(test_instance, all_scaler)
        cur_result = all_reg[s].predict(processed_test_instance)
        cur_result = cur_result.reshape(len(cur_result), 1)
        test_result = np.hstack((test_result, cur_result))
    rf_predicted_solver = []
    for i in range(len(test_result)) :
        rf_predicted_solver.append(solvers[np.argmax(test_result[i])])
        
    prediction = []
    test_instance_cluster = scaler_cluster.transform(feature.loc[test_ind, dim_reduction])
    for i in range(len(test_instance_cluster)):
        prediction.append(bestNSolver(centroid, test_instance_cluster[i], 5))
    result_rf = resultAnalysis(test_result, best_score.loc[test_ind], test_all_scores)
    print("Average score of the prediction without clustering: ", averageScore(test_all_scores, rf_predicted_solver))
    sum_rf += averageScore(test_all_scores, rf_predicted_solver)
    test_instance = all_scores.iloc[test_ind, :1]
    for i in range(len(rf_predicted_solver)):
        if not (rf_predicted_solver[i] in prediction[i]):
            if (solvers[np.argsort(-test_result[i])[1]] in prediction[i][:1]) and (test_result[i][np.argsort(-test_result[i])[0]] - test_result[i][np.argsort(-test_result[i])[1]] < 0.01):
                rf_predicted_solver[i] = solvers[np.argsort(-test_result[i])[1]]
            
            
    print("Percentage of correct best solver predicted: ", accuracy(test_best_solver, rf_predicted_solver))
    sum_cluster += averageScore(test_all_scores, rf_predicted_solver)
    print("Average score of the prediction: ", averageScore(test_all_scores, rf_predicted_solver))
    
    print(singleBestSolver(test_all_scores))
    
    print(oracleAveScore(best_score.iloc[test_ind, 1:]))
    

    
            
    result_cl = []
    for i in range(len(prediction)) :
        result_cl.append(np.amax([test_all_scores.iloc[i][prediction[i][0]]]))
    result_cluster = resultAnalysisCluster(best_score.loc[test_ind], result_cl)













