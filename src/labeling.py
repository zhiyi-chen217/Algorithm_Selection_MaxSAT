#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:40:13 2020

@author: chenzhiyi
"""

import pandas as pd
import numpy as np


def labelling(h_np, solver, s) :
    label = []
    if s == "score":
        for i in range(h_np.shape[0]):
            row = h_np[i][:]
            highestScore = np.amax(row)
            label.append(highestScore)
    elif s == "solver" :
        for i in range(h_np.shape[0]):
            row = h_np[i][:]
            highestScore = np.amax(row)
            label.append(list(solver[row == highestScore]))
    return label


h = pd.read_html("/Users/chenzhiyi/Desktop/HonoursProgram/experiments/unweighted_incomplete_300.html")
h = h[0]
h_noindex = h.loc[:, h.columns != "Benchmark"]
h.loc[:, h.columns != "Benchmark"] = h_noindex.applymap(lambda s: float(str(s).split(" ")[0]))
h = h.rename(columns={"Benchmark": "instance"})
h = h.set_index("instance")
h.to_csv("result_unweighted.csv")

h_np = h.to_numpy()

label = labelling(h_np, h.columns, 'score')

dicFrame = {'instance': h.index, 'best-score':label}
df = pd.DataFrame(data=dicFrame)
df = df.set_index('instance')
df.to_csv("per_instance_best_score_unweighted.csv")
    

    