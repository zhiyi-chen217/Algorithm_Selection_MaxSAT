import pandas as pd
import os
from itertools import chain
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
os.chdir("../../data_graph_color/original")

# train_result_file_name = "train_culb_data.csv"
# test_culb_file_name = "test_culb_data.csv"
test_dimacs_file_name = "test_dimacs_data.csv"
feature_file_name = "dimacs_and_culberson_features.csv"
algos = ["dsatur", "max_clique", "max_path", "max_degree", "min_width", "lex"]
test_result_dimacs_name = "predictive_experiments_test_dimacs.csv"
# test_result_culb_name = "predictive_experiments_test_culb.csv"
predicted_result_dimacs_55_name = "../../result/result_graph_color_dimacs_55.csv"
predicted_result_dimacs_52_name = "../../result/result_graph_color_dimacs_52.csv"
predicted_result_dimacs_01_name = "../../result/result_graph_color_dimacs_01.csv"
# predicted_result_culb_name = "../../result/result_graph_color_culb.csv"
# predicted_result_culb_01_name = "../../result/result_graph_color_culb_01.csv"
time_limit = 3600

def readCSV(fname):
    df = pd.read_csv(fname)
    df = df.fillna(0)
    return df

def count_result(algo, time_limit, result_df, all_instance):
    result_df = result_df.loc[result_df["time"] <= time_limit]
    cur_algo = result_df[result_df["experiment"].str.startswith(algo)]
    count = 0
    for i in all_instance:
        cur_algo_instance = cur_algo.loc[cur_algo["instance"] == i]
        cur_algo_instance = cur_algo_instance.loc[cur_algo_instance["optimal"]]
        if len(cur_algo_instance) > 0:
            count += 1
    return count

def count_predicted_result(result_algo, time_limit, df, all_instance):
    df = df.loc[df["time"] <= time_limit]
    count = 0
    for i in all_instance:
        algo = result_algo.loc[i, "predicted_algo"]
        cur_algo = df[df["experiment"].str.startswith(algo)]
        cur_algo_instance = cur_algo.loc[cur_algo["instance"] == i]
        cur_algo_instance = cur_algo_instance.loc[cur_algo_instance["optimal"]]
        if len(cur_algo_instance) > 0:
            count += 1
    return count

result_df = readCSV(test_result_dimacs_name)
result_algo_dimacs_55 = readCSV(predicted_result_dimacs_55_name)
result_algo_dimacs_52 = readCSV(predicted_result_dimacs_52_name)
result_algo_dimacs_01 = readCSV(predicted_result_dimacs_01_name)
# result_algo_culb = readCSV(predicted_result_culb_name)
# result_algo_culb_01 = readCSV(predicted_result_culb_01_name)
df = readCSV(test_dimacs_file_name)
all_instance = list(set(df.loc[:, "instance"]))
result_algo_dimacs_52 = result_algo_dimacs_52 .set_index("instance")
result_algo_dimacs_55 = result_algo_dimacs_55.set_index("instance")
result_algo_dimacs_01 = result_algo_dimacs_01.set_index("instance")
# result_algo_culb = result_algo_culb.set_index("instance")
# result_algo_culb_01 = result_algo_culb_01.set_index("instance")
algo = "C4.5DT"


solved_instances = []
predicted_solved_instances_52 = []
predicted_solved_instances_55 = []
# predicted_solved_instances = []
predicted_solved_instances_01 = []
y = []
time_limits = chain(range(0, 500, 50), range(500, 2000, 100), range(2000, 4000, 200))
for t in time_limits:
    count = count_result(algo, t, result_df, all_instance)
    # count_predicted_52 = count_predicted_result(result_algo_52, t, df, all_instance)
    # count_predicted_55 = count_predicted_result(result_algo_55, t, df, all_instance)
    count_predicted_55 = count_predicted_result(result_algo_dimacs_55, t, df, all_instance)
    count_predicted_52 = count_predicted_result(result_algo_dimacs_52, t, df, all_instance)
    count_predicted_01 = count_predicted_result(result_algo_dimacs_01, t, df, all_instance)
    predicted_solved_instances_55.append(count_predicted_55)
    predicted_solved_instances_52.append(count_predicted_52)
    predicted_solved_instances_01.append(count_predicted_01)
    solved_instances.append(count)
    # predicted_solved_instances_52.append(count_predicted_52)
    # predicted_solved_instances_55.append(count_predicted_55)
    y.append(t)

fig, ax = plt.subplots()
solved_instances = np.array(solved_instances) / len(all_instance)
predicted_solved_instances_52 = np.array(predicted_solved_instances_52) / len(all_instance)
predicted_solved_instances_55 = np.array(predicted_solved_instances_55) / len(all_instance)
# predicted_solved_instances = np.array(predicted_solved_instances) / len(all_instance)
predicted_solved_instances_01 = np.array(predicted_solved_instances_01) / len(all_instance)#Create a figure and an axes.
ax.plot(y, solved_instances, ".-", label='C4.5DT')
ax.plot(y, predicted_solved_instances_52, "*-", label="Predicted (40 predicates)")
ax.plot(y, predicted_solved_instances_55, "1-", label="Predicted (29 predicates)")
# Plot some data on the axes.
# ax.plot(y, predicted_solved_instances_55, "*-", label="Predicted: Time")
# ax.plot(y, predicted_solved_instances_52, "1-", label="Predicted: Time")
ax.plot(y, predicted_solved_instances_01, "2-", label="Predicted: Zero-One")
# ax.plot(predicates, gap_covered, label='testing')
ax.set_xlabel("Seconds")
ax.set_ylabel("Solved Instance %")
# ax.set_title("Training and Testing performance of decision tree with depth 2")
plt.legend()
plt.savefig("compare_graph_color_culb_all.pdf", format="pdf")
# plt.show()
print("end")
