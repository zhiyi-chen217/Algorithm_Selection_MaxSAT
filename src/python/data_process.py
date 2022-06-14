import pandas as pd
import os
import random

os.chdir("../../data")

feature_file_name = "feature_unweighted.csv"
result_file_name = "result_unweighted.csv"
best_score_file_name = "per_instance_best_score_unweighted.csv"
zero_one_file_name = "result_zero_one_unweighted.csv"
n_instance = 297

def readCSV(fname):
    index = [i for i in range(0, n_instance)]
    df = pd.read_csv(fname)
    df = df.fillna(0)
    df = df.sort_values(by=['instance'])
    df.insert(0, 'ind', index)
    df = df.set_index('ind')
    return df


feature_df = readCSV(feature_file_name)
result_df = readCSV(result_file_name)
oracle_df = readCSV(best_score_file_name)
zero_one_df = readCSV(zero_one_file_name)
index = [i for i in range(0, n_instance)]
random.shuffle(index)

feature_df = feature_df.iloc[index, :]
result_df = result_df.iloc[index, :]
oracle_df = oracle_df.iloc[index, :]
zero_one_df = zero_one_df.iloc[index, :]
feature_df.to_csv("feature_unweighted_shuffled.csv", index=False)
result_df.to_csv("result_unweighted_shuffled.csv", index=False)
oracle_df.to_csv("per_instance_best_score_unweighted_shuffled.csv", index=False)
zero_one_df.to_csv("result_zero_one_unweighted_shuffled.csv", index=False)


