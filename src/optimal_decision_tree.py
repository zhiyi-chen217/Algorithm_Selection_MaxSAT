import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from MurTree import PredicateNode, ClassificationNode, MurTree


predicates = [('sepal_length', lambda x: x > 5.84), ('sepal_width', lambda x: x > 3.054),
              ('petal_length', lambda x: x > 3.76), ('petal_width', lambda x: x > 1.2),
              ('sepal_length', lambda x: x > 2.42), ('sepal_width', lambda x: x > 1.5),
              ('petal_length', lambda x: x > 1.38), ('petal_width', lambda x: x > 0.6),
              ('sepal_length', lambda x: x > 7), ('sepal_width', lambda x: x > 3.7),
              ('petal_length', lambda x: x > 5), ('petal_width', lambda x: x > 1.8),
              ('sepal_length', lambda x: x > 6.5), ('sepal_width', lambda x: x > 3.2),
              ('petal_width', lambda x: x > 0.75), ('petal_length', lambda x: x > 4.75)]

def readCSV(fname, n_instance):
    index = [i for i in range(0, n_instance)]
    df = pd.read_csv(fname)
    df = df.sort_values(by=['instance'])
    df.insert(0, 'ind', index)
    df = df.set_index('ind')
    df = df.fillna(0)
    return df

def computeLoss(result, target):
    loss = 0
    for i in range(len(result)):
        if not result[i] == target[i]:
            loss += 1
    return loss


def zeros(axis_1, axis_0=1):
    if axis_0 == 1:
        return [0 for i in range(axis_1)]
    return [zeros(axis_1) for i in range(axis_0)]

def computeFQ(data):
    n_predicate = len(predicates)
    FQ_single_dict = {c: zeros(n_predicate) for c in classes}
    FQ_pair_dict = {c: zeros(n_predicate, n_predicate) for c in classes}
    FQ_all_dict = {c: len(data.loc[data["label"] == c]) for c in classes}
    for ind in data.index:
        d = data.loc[ind, :]
        FQ_single = FQ_single_dict[d.label]
        FQ_pair = FQ_pair_dict[d.label]
        for i in range(len(predicates)):
            (feature_i, predicate_i) = predicates[i]
            if predicate_i(d[feature_i]):
                FQ_single[i] += 1
            for j in range(i, len(predicates)):
                (feature_j, predicate_j) = predicates[j]
                if predicate_i(d[feature_i]) and predicate_j(d[feature_j]):
                    FQ_pair[i][j] += 1
    return FQ_single_dict, FQ_pair_dict, FQ_all_dict

def findDepthTwo(data):
    FQ_single_dict, FQ_pair_dict, FQ_all_dict = computeFQ(data)
    best_left_subtree_mis = {i: len(data) + 1 for i in range(len(predicates))}
    best_left_subtree_feature = {i: None for i in range(len(predicates))}
    best_left_subtree_left_label = {i: None for i in range(len(predicates))}
    best_left_subtree_right_label = {i: None for i in range(len(predicates))}

    best_right_subtree_mis = {i: len(data) + 1 for i in range(len(predicates))}
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
            loss_left_left = np.sum(cs_left_left) - cs_left_left[ind_left_left]

            ind_left_right = np.argmax(cs_left_right)
            label_left_right = classes[ind_left_right]
            loss_left_right = np.sum(cs_left_right) - cs_left_right[ind_left_right]

            loss_left = loss_left_left + loss_left_right

            ind_right_left = np.argmax(cs_right_left)
            label_right_left = classes[ind_right_left]
            loss_right_left = np.sum(cs_right_left) - cs_right_left[ind_right_left]

            ind_right_right = np.argmax(cs_right_right)
            label_right_right = classes[ind_right_right]
            loss_right_right = np.sum(cs_right_right) - cs_right_right[ind_right_right]

            loss_right = loss_right_left + loss_right_right

            if loss_left < best_left_subtree_mis[i]:
                best_left_subtree_mis[i] = loss_left
                best_left_subtree_feature[i] = predicates[j]
                best_left_subtree_left_label[i] = label_left_left
                best_left_subtree_right_label[i] = label_left_right
            if loss_right < best_right_subtree_mis[i]:
                best_right_subtree_mis[i] = loss_right
                best_right_subtree_feature[i] = predicates[j]
                best_right_subtree_left_label[i] = label_right_left
                best_right_subtree_right_label[i] = label_right_right
    total_loss = []
    for i in range(len(predicates)):
        total_loss.append(best_left_subtree_mis[i] + best_right_subtree_mis[i])
    ind_pred = np.argmin(total_loss)
    root = PredicateNode(predicates[ind_pred][0], predicates[ind_pred][1])
    left = PredicateNode(best_left_subtree_feature[ind_pred][0], best_left_subtree_feature[ind_pred][1])
    right = PredicateNode(best_right_subtree_feature[ind_pred][0], best_right_subtree_feature[ind_pred][1])
    left.setLeft(ClassificationNode(best_left_subtree_left_label[ind_pred]))
    left.setRight(ClassificationNode(best_left_subtree_right_label[ind_pred]))
    right.setLeft(ClassificationNode(best_right_subtree_left_label[ind_pred]))
    right.setRight(ClassificationNode(best_right_subtree_right_label[ind_pred]))
    root.setLeft(left)
    root.setRight(right)
    return root, total_loss[ind_pred]

def findOptimalTree(data, depth):
    if depth == 2:
        return findDepthTwo(data)
    min_loss = len(data) + 1
    min_left_subtree = None
    min_right_subtree = None
    root = None
    for i in range(len(predicates)):
        (feature, predicate) = predicates[i]
        right = predicate(data.loc[:, feature])
        left = ~right
        left_subtree, left_loss = findOptimalTree(data.loc[left, :], depth-1)
        right_subtree, right_loss = findOptimalTree(data.loc[right, :], depth-1)
        total_loss = left_loss + right_loss
        if total_loss < min_loss:
            min_loss = total_loss
            root = predicates[i]
            min_left_subtree = left_subtree
            min_right_subtree = right_subtree
    root_node = PredicateNode(root[0], root[1])
    root_node.setLeft(min_left_subtree)
    root_node.setRight(min_right_subtree)
    return root_node, min_loss



# n_node = 50
# max_depth = 10
# n_instance = 297
# feature = readCSV("../data/feature_extended_unweighted.csv", n_instance)
# best_score = readCSV("../data/per_instance_best_score_unweighted.csv", n_instance)
# results_solvers = readCSV("../data/result_unweighted.csv", n_instance)
# classes = list(results_solvers.columns)[1:]


iris_data = pd.read_csv('../../IRIS.csv')
iris_data = iris_data.rename(columns={'species': 'label'})
classes = list(iris_data['label'].unique())
rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=3)

for tr, te in rs.split(iris_data):
    train_ind = tr
    test_ind = te

root, train_loss = findOptimalTree(iris_data.iloc[train_ind, :], depth=3)
tree = MurTree(root)
result = []
for ind in iris_data.iloc[test_ind, :].index:
    result.append(tree.classify(iris_data.loc[ind, :]))
test_loss = computeLoss(result, list(iris_data.iloc[test_ind, :]['label']))
print(computeLoss(result, list(iris_data.iloc[test_ind, :]['label'])))

