import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from MurTree import PredicateNode, ClassificationNode, MurTree

os.chdir("../../data")

def readCSV(fname, n_instance):
    index = [i for i in range(0, n_instance)]
    df = pd.read_csv(fname)
    df = df.sort_values(by=['instance'])
    df = df.set_index('instance')
    df = df.fillna(0)
    return df

def computeLoss(result, target):
    loss = 0
    for i in range(len(result)):
        if not result[i] == target[i]:
            loss += 1
    return loss

def computeGain(instance_result, result_score):
    total_gain = 0
    for (instance, result) in instance_result:
        total_gain += result_score.loc[instance, result]
    return total_gain


def zeros(axis_1, axis_0=1):
    if axis_0 == 1:
        return [0 for i in range(axis_1)]
    return [zeros(axis_1) for i in range(axis_0)]

def computeFQ(data, result):
    n_predicate = len(predicates)
    FQ_single_dict = {c: zeros(n_predicate) for c in classes}
    FQ_pair_dict = {c: zeros(n_predicate, n_predicate) for c in classes}
    FQ_all_dict = {c: result.loc[:, c].sum() for c in classes}
    for c in classes:
        FQ_single = FQ_single_dict[c]
        FQ_pair = FQ_pair_dict[c]
        for ind in data.index:
            d = data.loc[ind, :]
            for i in range(len(predicates)):
                (feature_i, predicate_i, threshold_i) = predicates[i]
                if predicate_i(d[feature_i]):
                    FQ_single[i] += result.loc[ind, c]
                for j in range(i, len(predicates)):
                    (feature_j, predicate_j, threshold_j) = predicates[j]
                    if predicate_i(d[feature_i]) and predicate_j(d[feature_j]):
                        FQ_pair[i][j] += result.loc[ind, c]
    return FQ_single_dict, FQ_pair_dict, FQ_all_dict

def findDepthTwo(data, result):
    FQ_single_dict, FQ_pair_dict, FQ_all_dict = computeFQ(data, result)
    best_left_subtree_gain = {i: -1 for i in range(len(predicates))}
    best_left_subtree_feature = {i: None for i in range(len(predicates))}
    best_left_subtree_left_label = {i: None for i in range(len(predicates))}
    best_left_subtree_right_label = {i: None for i in range(len(predicates))}

    best_right_subtree_gain = {i: -1 for i in range(len(predicates))}
    best_right_subtree_feature = {i: None for i in range(len(predicates))}
    best_right_subtree_left_label = {i: None for i in range(len(predicates))}
    best_right_subtree_right_label = {i: None for i in range(len(predicates))}
    for i in range(len(predicates)):
        for j in range(len(predicates)):
            if i == j:
                continue
            cs_left_left = []
            cs_left_right = []
            cs_right_left = []
            cs_right_right = []
            for c in classes:
                FQ_c = FQ_all_dict[c]
                FQ_c_i = FQ_single_dict[c][i]
                FQ_c_j = FQ_single_dict[c][j]
                if j > i:
                    FQ_c_i_j = FQ_pair_dict[c][i][j]
                else:
                    FQ_c_i_j = FQ_pair_dict[c][j][i]
                cs_left_left.append(FQ_c - FQ_c_i - FQ_c_j + FQ_c_i_j)
                cs_left_right.append(FQ_c_j - FQ_c_i_j)
                cs_right_left.append(FQ_c_i - FQ_c_i_j)
                cs_right_right.append(FQ_c_i_j)
            ind_left_left = np.argmax(cs_left_left)
            label_left_left = classes[ind_left_left]
            gain_left_left = cs_left_left[ind_left_left]

            ind_left_right = np.argmax(cs_left_right)
            label_left_right = classes[ind_left_right]
            gain_left_right = cs_left_right[ind_left_right]

            gain_left = gain_left_left + gain_left_right

            ind_right_left = np.argmax(cs_right_left)
            label_right_left = classes[ind_right_left]
            gain_right_left = cs_right_left[ind_right_left]

            ind_right_right = np.argmax(cs_right_right)
            label_right_right = classes[ind_right_right]
            gain_right_right = cs_right_right[ind_right_right]

            gain_right = gain_right_left + gain_right_right

            if gain_left > best_left_subtree_gain[i]:
                best_left_subtree_gain[i] = gain_left
                best_left_subtree_feature[i] = predicates[j]
                best_left_subtree_left_label[i] = label_left_left
                best_left_subtree_right_label[i] = label_left_right
            if gain_right > best_right_subtree_gain[i]:
                best_right_subtree_gain[i] = gain_right
                best_right_subtree_feature[i] = predicates[j]
                best_right_subtree_left_label[i] = label_right_left
                best_right_subtree_right_label[i] = label_right_right
    total_gain = []
    for i in range(len(predicates)):
        total_gain.append(best_left_subtree_gain[i] + best_right_subtree_gain[i])
    ind_pred = np.argmax(total_gain)
    root = PredicateNode(predicates[ind_pred][0], predicates[ind_pred][1], predicates[ind_pred][2])
    left = PredicateNode(best_left_subtree_feature[ind_pred][0], best_left_subtree_feature[ind_pred][1],
                         best_left_subtree_feature[ind_pred][2])
    right = PredicateNode(best_right_subtree_feature[ind_pred][0], best_right_subtree_feature[ind_pred][1],
                          best_right_subtree_feature[ind_pred][2])
    left.setLeft(ClassificationNode(best_left_subtree_left_label[ind_pred]))
    left.setRight(ClassificationNode(best_left_subtree_right_label[ind_pred]))
    right.setLeft(ClassificationNode(best_right_subtree_left_label[ind_pred]))
    right.setRight(ClassificationNode(best_right_subtree_right_label[ind_pred]))
    root.setLeft(left)
    root.setRight(right)
    return root, total_gain[ind_pred]

def findOptimalTree(data, result, depth):
    if depth == 2:
        return findDepthTwo(data, result)
    min_gain = -1
    min_left_subtree = None
    min_right_subtree = None
    root = None
    for i in range(len(predicates)):
        (feature, predicate, threshold) = predicates[i]
        right = predicate(data.loc[:, feature])
        left = ~right
        left_subtree, left_gain = findOptimalTree(data.loc[left, :], result.loc[left, :], depth-1)
        right_subtree, right_gain = findOptimalTree(data.loc[right, :], result.loc[right, :], depth-1)
        total_gain = left_gain + right_gain
        if total_gain > min_gain:
            min_gain = total_gain
            root = predicates[i]
            min_left_subtree = left_subtree
            min_right_subtree = right_subtree
    root_node = PredicateNode(root[0], root[1], root[2])
    root_node.setLeft(min_left_subtree)
    root_node.setRight(min_right_subtree)
    return root_node, min_gain

def createLambda(limit):
    return lambda x: x > limit


def constructPredicates(max_pred):
    feature_col_map = {feature.columns[i]: i for i in range(feature.columns.size)}
    features = list(feature.columns)
    predicates = []
    predicates_return = []
    for c in classes:
        target = results_solvers.iloc[train_ind, :][c]
        reg = DecisionTreeRegressor()
        reg.fit(feature.iloc[train_ind, :], target)
        for t in list(zip(reg.tree_.feature, reg.tree_.threshold)):
            if t[0] >= 0:
                predicates.append([t[0], t[1]])
                predicates_return.append((features[t[0]], createLambda(t[1]), t[1]))
    for c in classes[4:4]:
        target = results_solvers.iloc[train_ind, :][c]
        reg = DecisionTreeRegressor()
        reg.fit(feature_reduction.iloc[train_ind, :], target)
        features_reducted = list(feature_reduction.columns)
        for t in list(zip(reg.tree_.feature, reg.tree_.threshold)):
            if t[0] >= 0:
                predicates.append([feature_col_map[features_reducted[t[0]]], t[1]])
    random.shuffle(predicates)
    pred_df = pd.DataFrame(predicates[:max_pred], columns=['feature', 'threshold'])
    pred_df = pred_df.set_index('feature')
    pred_df.to_csv("../src/c++/predicate.csv")
    return predicates_return[:max_pred]
    # return predicates[:min(max_pred, len(predicates))]

n_node = 50
max_depth = 10
n_instance = 297
feature = readCSV("feature_unweighted_shuffled.csv", n_instance)
feature_reduction = readCSV("feature_reduction_unweighted.csv", n_instance)
feature_reduction_lasso = readCSV("feature_reduction_lasso_unweighted.csv", n_instance)
best_score = readCSV("per_instance_best_score_unweighted_shuffled.csv", n_instance)
results_solvers = readCSV("result_unweighted_shuffled.csv", n_instance)
classes = list(results_solvers.columns)

rs = ShuffleSplit(n_splits=1, test_size=0.25)

for tr, te in rs.split(feature):
    train_ind = tr
    test_ind = te

# test_size = int(n_instance / 5)
# i = 0
# start = test_size * i
# end = test_size * (i + 1)
# train_ind = [j for j in range(0, start)] + [j for j in range(end, n_instance)]
# test_ind = [j for j in range(start, end)]

#predicates = constructPredicates(500)

predicates = []
for i in range(feature.columns.size):
    mu = feature.iloc[:, i].mean()
    std = feature.iloc[:, i].std()
    if std > 0:
        t = np.quantile(feature.iloc[:, i], np.linspace(0, 1, 20))
        for temp in t:
            predicates.append([i, temp])
random.shuffle(predicates)
pred_df = pd.DataFrame(predicates[:], columns=['feature', 'threshold'])
pred_df = pred_df.set_index('feature')
pred_df.to_csv("../src/c++/predicate.csv")


# root, train_gain = findOptimalTree(feature.iloc[train_ind, :], results_solvers.iloc[train_ind, :], depth=2)
#
# tree = MurTree(root)
# result = []
# for ind in feature.iloc[test_ind, :].index:
#     result.append((ind, tree.classify(feature.loc[ind, :])))
# test_gain = computeGain(result, results_solvers.iloc[test_ind, :]) / len(test_ind)
# SBS_gain = results_solvers.iloc[test_ind, :].loc[:, 'Loandra'].sum() / len(test_ind)
#
# train_result = []
# for ind in feature.iloc[train_ind, :].index:
#     train_result.append((ind, tree.classify(feature.loc[ind, :])))
# train__gain = computeGain(train_result, results_solvers.iloc[train_ind, :])
# SBS_train_gain = results_solvers.iloc[train_ind, :].loc[:, 'Loandra'].sum()
#
# tree.drawTree()

