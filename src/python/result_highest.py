import pandas as pd
import os
import random

os.chdir("../../data")

feature_file_name = "feature_unweighted.csv"
result_file_name = "result_unweighted.csv"
best_score_file_name = "per_instance_best_score_unweighted.csv"
n_instance = 297

def readCSV(fname):
    df = pd.read_csv(fname)
    df = df.fillna(0)
    df = df.sort_values(by=['instance'])
    df = df.set_index('instance')
    return df


feature_df = readCSV(feature_file_name)
result_df = readCSV(result_file_name)
oracle_df = readCSV(best_score_file_name)
true_false_df = pd.DataFrame(index=feature_df.index, columns=result_df.columns)
true_false_df = true_false_df.fillna(0)
for i in result_df.index:
    best_score = oracle_df.loc[i, 'best-score']
    for s in result_df.columns:
        cur_score = result_df.loc[i, s]
        if cur_score == best_score:
            true_false_df.loc[i, s] = 1
true_false_df.to_csv("result_zero_one_unweighted.csv")
print("end")

