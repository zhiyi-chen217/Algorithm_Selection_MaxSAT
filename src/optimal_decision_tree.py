import numpy as np
import pandas as pd

predicates = [i for i in range(785)]
n_node = 50
max_depth = 10
data_set = pd.read_csv("../../train.csv")
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def zeros(axis_1, axis_0=1):
    if axis_0 == 1:
        return [0 for i in range(axis_1)]
    return [zeros(axis_1) for i in range(axis_0)]

def findDepthTwo(data):
    n_feature = len(data.columns) - 1
    FQ_single_dict = {c: zeros(n_feature) for c in classes}
    FQ_pair_dict = {c: zeros(n_feature, n_feature) for c in classes}
    FQ_all_dict = {c: len(data.loc[data_set["label"] == c]) for c in classes}
    features = data.columns[1:]
    for ind in data.index:
        d = data.loc[ind, :]
        FQ_single = FQ_single_dict[d.label]
        FQ_pair = FQ_pair_dict[d.label]
        for i in range(len(features)):
            if d[features[i]] > 100:
                FQ_single[i] += 1
            for j in range(i, len(features)):
                if d[features[i]] > 100 and d[features[j]] > 100:
                    FQ_pair[i][j] += 1
    FQ_all_dict = {c: len(data.loc[data_set["label"] == c]) for c in classes}
    best_left_subtree_mis = {i: len(data) + 1 for i in range(len(features))}
    best_left_subtree_feature = {i: None for i in range(len(features))}
    best_left_subtree_left_label = {i: None for i in range(len(features))}
    best_left_subtree_right_label = {i: None for i in range(len(features))}

    best_right_subtree_mis = {i: len(data) + 1 for i in range(len(features))}
    best_right_subtree_feature = {i: None for i in range(len(features))}
    best_right_subtree_left_label = {i: None for i in range(len(features))}
    best_right_subtree_right_label = {i: None for i in range(len(features))}
    for i in range(len(features)):
        for j in range(len(features)):
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
                FQ_c_i_j = FQ_pair_dict[c][i][j]
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
                best_left_subtree_feature[i] = features[j]
                best_left_subtree_left_label[i] = label_left_left
                best_left_subtree_right_label[i] = label_left_right
            if loss_right < best_right_subtree_mis[i]:
                best_right_subtree_mis[i] = loss_right
                best_right_subtree_feature[i] = features[j]
                best_right_subtree_left_label[i] = label_right_left
                best_right_subtree_right_label[i] = label_right_right
    total_loss = []
    for i in range(len(features)):
        total_loss.append(best_left_subtree_mis[i] + best_right_subtree_mis[i])
    ind_feature = np.argmin(total_loss)
    root = features[ind_feature]
    left = best_left_subtree_feature[ind_feature]
    right = best_right_subtree_feature[ind_feature]
    return root, left, right


findDepthTwo(data_set.iloc[:50, :])
