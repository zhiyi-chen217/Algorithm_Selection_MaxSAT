import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
os.chdir("../../result")
results = pd.read_csv("result_3_graph.csv")
start = 4
end = 100
step = 5

predicates = [i for i in range(start, end, step)]

solved = np.array(results.loc[:, "Solved"]) / 137
# gain = np.array(results.loc[:, "Gain"])
# gap_covered = np.array(results.loc[:, "Average Gap Covered"])
# gain = (gain - np.mean(gain)) / np.std(gain)
# gap_covered = (gap_covered - np.mean(gap_covered)) / np.std(gap_covered)

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(predicates, solved, label='test')  # Plot some data on the axes.
# ax.plot(predicates, gap_covered, label='testing')
ax.set_xlabel("#Predicates")
ax.set_ylabel("Solved Instance%")
# ax.set_title("Training and Testing performance of decision tree with depth 2")
plt.legend()
plt.savefig("test_curve_graph_3.pdf", format="pdf")
# plt.show()
print("end")
