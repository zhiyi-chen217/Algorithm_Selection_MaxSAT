import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

os.chdir("../../data_graph_color")

train_feature_file_name = "train_feature.csv"
train_result_file_name = "all_ordering_zero_one_train_result.csv"
train_time_file_name = "all_ordering_time_train_result.csv"

algos = ["dsatur", "max_clique", "max_path", "max_degree", "min_width", "lex"]

def readCSV(fname):
    df = pd.read_csv(fname)
    df = df.sort_values(by=['instance'])
    df = df.set_index('instance')
    df = df.fillna(0)
    return df

train_feature_df = readCSV(train_feature_file_name)
train_result_df = readCSV(train_result_file_name)


clf = DecisionTreeClassifier()
reg = DecisionTreeRegressor()
predicates = []
for algo in algos:
    clf.fit(train_feature_df.iloc[:, 1:], train_result_df.loc[:, algo])
    reg.fit(train_feature_df.iloc[:, 1:], train_result_df.loc[:, algo])
    for t in list(zip(clf.tree_.feature, clf.tree_.threshold)):
        if t[0] >= 0:
            predicates.append([t[0], t[1]])
    for t in list(zip(reg.tree_.feature, reg.tree_.threshold)):
        if t[0] >= 0:
            predicates.append([t[0], t[1]])
random.shuffle(predicates)
pred_df = pd.DataFrame(predicates[:250], columns=['feature', 'threshold'])
pred_df = pred_df.set_index('feature')
pred_df.to_csv("../src/c++/predicate_graph_color.csv")
print("end")
