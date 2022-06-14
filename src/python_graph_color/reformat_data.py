import pandas as pd
import os

os.chdir("../../data_graph_color/original")

train_result_file_name = "train_culb_data.csv"
test_culb_file_name = "test_culb_data.csv"
test_dimacs_file_name = "test_dimacs_data.csv"
feature_file_name = "dimacs_and_culberson_features.csv"
algos = ["dsatur", "max_clique", "max_path", "max_degree", "min_width", "lex"]
time_limit = 3600

def readCSV(fname, time_limit):
    df = pd.read_csv(fname)
    df = df.loc[df["time"] <= time_limit]
    df = df.fillna(0)
    return df

def reformat_result(file_name, result_file_name):
    df = readCSV(file_name, time_limit)
    all_train_instance = list(set(df.loc[:, "instance"]))
    df_save = pd.DataFrame(index=all_train_instance, columns=algos)

    for algo in algos:
        cur_algo = df[df["experiment"].str.startswith(algo)]
        for i in all_train_instance:
            cur_algo_instance = cur_algo.loc[cur_algo["instance"] == i]
            cur_algo_instance = cur_algo_instance.loc[cur_algo_instance["optimal"]]
            if len(cur_algo_instance) > 0:
                df_save.loc[i, algo] = 1
            else:
                df_save.loc[i, algo] = 0
    df_save = df_save.sort_index()
    df_save.to_csv(result_file_name, index_label="instance")

def reformat_result_time(file_name, result_file_name):
    df = readCSV(file_name, time_limit)
    all_train_instance = list(set(df.loc[:, "instance"]))
    df_save = pd.DataFrame(index=all_train_instance, columns=algos)

    for algo in algos:
        cur_algo = df[df["experiment"].str.startswith(algo)]
        for i in all_train_instance:
            cur_algo_instance = cur_algo.loc[cur_algo["instance"] == i]
            cur_algo_instance = cur_algo_instance.loc[cur_algo_instance["optimal"]]
            if len(cur_algo_instance) > 0:
                df_save.loc[i, algo] = time_limit - cur_algo_instance.loc[:, "time"].iloc[0]
            else:
                df_save.loc[i, algo] = 0
    df_save = df_save.sort_index()
    df_save.to_csv(result_file_name, index_label="instance")

def reformat_feature():
    feature_df = pd.read_csv(feature_file_name)
    feature_df = feature_df.fillna(0)
    train_result_df = pd.read_csv(train_result_file_name)
    test_culb_df = pd.read_csv(test_culb_file_name)
    test_dimacs_df = pd.read_csv(test_dimacs_file_name)

    all_train_instance = list(set(train_result_df.loc[:, "instance"]))
    test_culb_instance = list(set(test_culb_df.loc[:, "instance"]))
    test_dimacs_instance = list(set(test_dimacs_df.loc[:, "instance"]))

    train_feature_df = feature_df.loc[feature_df["instance"].isin(all_train_instance)]
    train_feature_df = train_feature_df.sort_values(["instance"])
    train_feature_df.to_csv("../train_feature.csv", index=False)

    test_culb_feature_df = feature_df.loc[feature_df["instance"].isin(test_culb_instance)]
    test_culb_feature_df = test_culb_feature_df.sort_values(["instance"])
    test_culb_feature_df.to_csv("../test_culb_feature.csv", index=False)

    test_dimacs_feature_df = feature_df.loc[feature_df["instance"].isin(test_dimacs_instance)]
    test_dimacs_feature_df = test_dimacs_feature_df.sort_values(["instance"])
    test_dimacs_feature_df.to_csv("../test_dimacs_feature.csv", index=False)

reformat_result_time(train_result_file_name, "../all_ordering_time_train_result.csv")
# reformat_feature()
print("end")
