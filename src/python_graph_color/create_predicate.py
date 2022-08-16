import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

os.chdir("../../data_graph_color")

train_feature_file_name = "train_feature.csv"
train_result_file_name = "all_ordering_zero_one_train_result.csv"
train_time_file_name = "all_ordering_time_train_result.csv"
test_feature_file_name = "test_dimacs_feature.csv"
test_result_file_name = "all_ordering_zero_one_test_dimacs_result.csv"

algos = ["dsatur", "max_clique", "max_path", "max_degree", "min_width", "lex"]

parameter_space_grid = {
    'RandomForestRegressor' : {
        'n_estimators' : [i for i in range(10, 60, 2)],
        'min_samples_leaf' : [i for i in range(1, 10, 2)],
        'min_samples_split' : [i for i in range(2, 20, 2)]
        },
    'DecisionTreeRegressor': {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_depth': [3, 4, 5, 8, 10, 15],
        'min_samples_split': [i for i in range(2, 10, 2)],
        'min_samples_leaf': [i for i in range(1, 10, 2)],
    },
    'DecisionTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': [3, 4, 5, 8, 10, 15],
        'min_samples_split': [i for i in range(2, 10, 2)],
        'min_samples_leaf': [i for i in range(1, 10, 2)],
    }
}

def readCSV(fname):
    df = pd.read_csv(fname)
    df = df.sort_values(by=['instance'])
    df = df.set_index('instance')
    df = df.fillna(0)
    return df

def solved_score(scores, predicted_scores, algos) :
    predict_algos = []
    solved_score = 0
    for i in range(len(scores)):
        predicted_algo = algos[np.argmax(predicted_scores[i])]
        predict_algos.append(predicted_algo)
        if scores.iloc[i, :][predicted_algo] == 1:
            solved_score += 1
    return solved_score

train_feature_df = readCSV(train_feature_file_name)
train_result_df = readCSV(train_result_file_name)
test_feature_df = readCSV(test_feature_file_name)
test_result_df = readCSV(test_result_file_name)

clfs = {}
# reg = DecisionTreeRegressor()
predicates = []
for algo in algos:
    clf = DecisionTreeClassifier()
    clf = GridSearchCV(clf, parameter_space_grid['DecisionTreeClassifier'], refit=True)
    clf.fit(train_feature_df.iloc[:, :], train_result_df.loc[:, algo])
    clfs[algo] = clf
    # reg.fit(train_feature_df.iloc[:, 1:], train_result_df.loc[:, algo])
    for t in list(zip(clf.best_estimator_.tree_.feature, clf.best_estimator_.tree_.threshold)):
        if t[0] >= 0:
            predicates.append([t[0], t[1]])
    # for t in list(zip(reg.tree_.feature, reg.tree_.threshold)):
    #     if t[0] >= 0:
    #         predicates.append([t[0], t[1]])

test_result = np.array([[] for i in range(len(test_feature_df))])
for algo in algos:
    cur_result = clfs[algo].predict(test_feature_df)
    cur_result = cur_result.reshape(len(cur_result), 1)
    test_result = np.hstack((test_result, cur_result))
n_solved = solved_score(test_result_df, test_result, algos)
random.shuffle(predicates)
pred_df = pd.DataFrame(predicates, columns=['feature', 'threshold'])
pred_df = pred_df.set_index('feature')
pred_df.to_csv("../src/c++/predicate_graph_color_reg.csv")
print("end")
