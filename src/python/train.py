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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import export_text
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, Birch
from scipy.special import softmax
from random import choices
import os
import ast


os.chdir("../../data")

feature_file_name = "feature_more_balance_feature_weighted.csv"
result_file_name = "result_weighted.csv"
best_solver_file_name = "per_instance_best_solver_weighted.csv"
best_score_file_name = "per_instance_best_score_weighted.csv"
problem_label_file_name = "problem_label.csv"
unique_value_limit = 50
n_instance = 297
n_cluster = 4
sum_cluster = 0
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

def averageScore(actual_scores, solvers) :
    result = 0
    for i in range(len(solvers)) :
        result = result + actual_scores.iloc[i][solvers[i]]
    return result/len(solvers)

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

    
instance_features = ['instance', 'ncls', 'nhard_len_stats.ave',
       'nhard_len_stats.max', 'nhard_len_stats.min', 'nhard_len_stats.stddev',
       'nhards', 'nsoft_len_stats.ave', 'nsoft_len_stats.max',
       'nsoft_len_stats.min', 'nsoft_len_stats.stddev', 'nsoft_wts', 'nsofts',
       'nvars', 'soft_wt_stats.ave', 'soft_wt_stats.max', 'soft_wt_stats.min',
       'soft_wt_stats.stddev']
balance_features = ['balance-hard', 'balance-hard-max', 'balance-hard-mean',
       'balance-hard-min', 'balance-hard-std','balance-soft',
       'balance-soft-max', 'balance-soft-mean', 'balance-soft-min',
       'balance-soft-std']
graph_features = ['VG-mean',
       'VG-max', 'VG-min', 'VG-std',
       'VCG-max', 'VCG-mean', 'VCG-min', 'VCG-std']
horn_features = ['Horn-fraction',
       'Horn-V-mean', 'Horn-V-max', 'Horn-V-min', 'Horn-V-std']
extra = ['balance-hard-entropy', 'balance-soft-entropy', 'VG-entropy']
all_features = instance_features + balance_features
used_features = all_features
feature = readCSV(feature_file_name)
feature = feature.loc[:,  used_features]

feature = feature.fillna(0)
column_selection = (feature.nunique() > unique_value_limit)
feature = feature.loc[:,  column_selection]
dim_reduction = feature.columns[1:]
#feature = expandFeature(feature)

all_scores = readCSV(result_file_name)
best_solver = readCSV(best_solver_file_name)
best_score = readCSV(best_score_file_name)
problem_label = readCSV(problem_label_file_name)

for i in range(n_iteration):
    rs = StratifiedShuffleSplit(n_splits=1, random_state=np.random.randint(40), test_size=0.25)
    
    for tr, te in rs.split(feature, problem_label.iloc[:, 1]):
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
    skf = StratifiedKFold(n_splits=5)
    
    # supervised learning
    for s in solvers:
        target = all_scores.loc[train_ind, s]
        
        reg = DecisionTreeRegressor()
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
    
    # prediction of probabilities
    output_prob = all_scores.iloc[train_ind, 1:]
    output_prob = softmax(output_prob, axis=1)
    reg_prob = linear_model.Ridge(alpha=0.5)
    reg_prob.fit(inputInstance, output_prob)
    
    # predict using model
    train_all_scores = all_scores.iloc[train_ind, 1:]
    test_instance = feature.iloc[test_ind, 1:]
    test_best_solver = best_solver.loc[test_ind, 'best-solver(s)']
    test_all_scores = all_scores.iloc[test_ind, 1:]
    test_result = np.array([[] for i in range(len(test_instance))])
    for s in solvers:
        processed_test_instance = preprocessing(test_instance, all_scaler)
        cur_result = all_reg[s].predict(processed_test_instance)
        cur_result = cur_result.reshape(len(cur_result), 1)
        test_result = np.hstack((test_result, cur_result))
        
    # predict using prob
    test_prob = reg_prob.predict(processed_test_instance)
    # collect best solver
    rf_predicted_solver = []
    prob_predicted_solver = []
    prob_predicted_solver_distribution = []
    for i in range(len(test_result)) :
        rf_predicted_solver.append(solvers[np.argmax(test_result[i])])
        probs =  softmax(test_prob[i]*500)
        prob_predicted_solver_distribution.append(choices(solvers, probs)[0])
        prob_predicted_solver.append(solvers[np.argmax(test_prob[i])])
        
    prediction = []
    test_instance_cluster = scaler_cluster.transform(feature.loc[test_ind, dim_reduction])
    for i in range(len(test_instance_cluster)):
        prediction.append(bestNSolver(centroid, test_instance_cluster[i], 2))
    result_rf = resultAnalysis(test_result, best_score.loc[test_ind], test_all_scores)
    test_instance = all_scores.iloc[test_ind, :1]
    # for i in range(len(rf_predicted_solver)):
    #     if (not solvers[np.argsort(-test_result[i])[0]] == prob_predicted_solver[i]) and (solvers[np.argsort(-test_result[i])[1]] ==  prob_predicted_solver[i]):
    #         print(test_instance.iloc[i, 0])
            
            
    print("Percentage of correct best solver predicted: ", accuracy(test_best_solver, rf_predicted_solver))
    sum_rf += averageScore(test_all_scores, rf_predicted_solver)
    print("Average score of the prediction: ", averageScore(test_all_scores, rf_predicted_solver))
    sum_prob += averageScore(test_all_scores, prob_predicted_solver)
    print("Average score of predicting probability: ", averageScore(test_all_scores, prob_predicted_solver))
    sum_prob_distribution += averageScore(test_all_scores, prob_predicted_solver_distribution)
    print("Average score of predicting probability (distribution): ", averageScore(test_all_scores, prob_predicted_solver_distribution))

    sum_sb += singleBestSolver(test_all_scores)
    print("Score of the single best solver:", singleBestSolver(test_all_scores))
    sum_oracle += oracleAveScore(best_score.iloc[test_ind, 1:])
    print("Score of the oracle:", oracleAveScore(best_score.iloc[test_ind, 1:]))
    

    
            
    result_cl = []
    result_prob = []
    for i in range(len(prediction)) :
        result_cl.append(np.amax([test_all_scores.iloc[i][prediction[i][0]]]))
        result_prob.append(np.amax([test_all_scores.iloc[i][prob_predicted_solver[i]]]))
    result_cluster = resultAnalysisCluster(best_score.loc[test_ind], result_cl)
    result_probability = resultAnalysisCluster(best_score.loc[test_ind], result_prob)
    for i in range(len(result_probability)):
        if (result_probability.iloc[i, 1] < result_rf.iloc[i, 1]):
            if result_probability.iloc[i, 0] in special_instance.keys():
                special_instance[result_probability.iloc[i, 0]] += 1
            else:
                 special_instance[result_probability.iloc[i, 0]] = 1


print("Average score of all predictions: ", sum_rf / n_iteration)
print("Average score of all predicting probability: ", sum_prob / n_iteration)
print("Average score of all predicting probability (distribution): ", sum_prob_distribution / n_iteration)
print("Average score of the single best solver:", sum_sb / n_iteration)
print("Average score of the oracle:", sum_oracle / n_iteration)










