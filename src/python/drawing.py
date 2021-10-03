import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir("../../result")
results = pd.read_csv("result_reduction.csv")
start = 4
end = 760
step = 3
predicates = [i for i in range(start, end, step)]

gain = np.array(results.loc[:, "Gain"])
gap_covered = np.array(results.loc[:, "Average Gap Covered"])
gain = (gain - np.mean(gain)) / np.std(gain)
gap_covered = (gap_covered - np.mean(gap_covered)) / np.std(gap_covered)

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(predicates, gain, label='training')  # Plot some data on the axes.
ax.plot(predicates, gap_covered, label='testing')
ax.set_xlabel("#predicate")
ax.set_ylabel("standardized performance")
# ax.set_title("Training and Testing performance of decision tree with depth 2")
plt.legend()
plt.savefig("train_test_curve_reduction.png")
plt.show()
