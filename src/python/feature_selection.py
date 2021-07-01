#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:40:16 2021

@author: chenzhiyi
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from itertools import chain, combinations
import os
import math
os.chdir("../data")
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

def selectFeature(feature, all_scores, expand_n):
    solvers = list(all_scores.columns)
    solvers.remove('instance')
    lambda1 = 0.0001
    input_instance = feature.iloc[:, 1:]
    input_instance["bias"] = np.ones(len(input_instance))
    all_solver_features = {}
    
    
    
    for s in solvers:
        print("solver:{}".format(s))
        target = all_scores.loc[:, s]
        scaler = StandardScaler()
        inputs = scaler.fit_transform(input_instance)
        
        reg_lasso = linear_model.Lasso(alpha=lambda1, max_iter=10000000)
        reg_lasso.fit(inputs, target)
        print("lasso start: {}".format(reg_lasso.score(inputs, target)))
        reg_rf = RandomForestRegressor()
        reg_rf.fit(inputs, target)
        print("random forest start: {}".format(reg_rf.score(inputs, target)))
        
        
        S = np.array(input_instance.columns)[np.array(list(map(lambda f: not math.isclose(f, 0), reg_lasso.coef_)))]
        X_s = input_instance.loc[:, S]
        X_new = expandFeature(X_s, expand_n)
        print(len(X_new.columns))
        scaler = StandardScaler()
        inputs_new = scaler.fit_transform(X_new)
        
        reg_lasso.fit(inputs_new, target)
        print("lasso after expand: {}".format(reg_lasso.score(inputs_new, target)))
        reg_rf.fit(inputs_new, target)
        print("random forest after expand: {}".format(reg_rf.score(inputs_new, target)))
        
        
        w_ridge = linear_model.Ridge(alpha=0.001).fit(inputs_new, target).coef_
        S_new_ = np.array(X_new.columns)[np.array(list(map(lambda f: not math.isclose(f, 0), w_ridge)))]
        X_new_ = X_new.loc[:, S_new_]
        scaler = StandardScaler()
        inputs_new_ = scaler.fit_transform(X_new_)
        
        w_ridge_ = w_ridge[[not math.isclose(w, 0) for w in w_ridge]]
        w_lasso = reg_lasso.fit(inputs_new_ / w_ridge_, target).coef_
        
        S_final = np.array(X_new_.columns)[np.array(list(map(lambda f: not math.isclose(f, 0), w_lasso)))]
        X_final = X_new_.loc[:, S_final]
        scaler = StandardScaler()
        inputs_final = scaler.fit_transform(X_final)
        reg_lasso.fit(inputs_final, target)
        print("lasso end: {}".format(reg_lasso.score(inputs_final, target)))
        reg_rf.fit(inputs_final, target)
        print("random forest end: {}".format(reg_rf.score(inputs_final, target)))
        
        all_solver_features[s] = S_final
    return all_solver_features
    


