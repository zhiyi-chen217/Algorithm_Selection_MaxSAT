#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:12:39 2021

@author: chenzhiyi
"""
import os
import gzip
import json
import numpy as np
import pandas
from pandas.io.json import json_normalize



def calculateFeature(nodes, prefix):
    result_dic = {}
    result_dic[prefix + "-" + 'mean'] = np.mean(nodes)
    result_dic[prefix + "-" + 'max'] = np.max(nodes)
    result_dic[prefix + "-" + 'min'] = np.min(nodes)
    result_dic[prefix + "-" + 'std'] = np.std(nodes)
    return result_dic



def addHornEdge(nodes, clause, count):
    clause = np.array(list(map(lambda c: int(c), clause)))
    n_positive = (clause > 0).sum()
    if n_positive <= 1:
        count = count + 1
        clause = np.array(list(map(lambda c: abs(c), clause)))
        for n in clause:
            nodes[n - 1] += 1
    return count


common = ['mean', 'max', 'min', 'std']
HV = list(map(lambda s: 'Horn-V-' + s, common))
all_feature = ['instance', 'Horn-fraction'] + HV

d = pandas.DataFrame(data=[], columns=all_feature)


os.chdir("/Users/chenzhiyi/Desktop/HonoursProgram/"
         "weighted_maxsat_instances/ms_evals/MS19/mse19-incomplete-weighted-benchmarks/")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".gz"):
            count = 0
            fname = os.path.join(root, file)
            print(fname)
            f = gzip.open(fname, 'rt')
            while True:
                s = f.readline()
                if s.startswith("p "):
                    break
            num = s.split(" ")
            nv = int(num[2])
            nc = int(num[3])
            V_nodes = np.zeros(nv)

            for i in range(nc):
                s = f.readline()
                if s.startswith("c"):
                    continue
                clause = s.split(" ")
                clause = clause[1:len(clause) - 1]
                count = addHornEdge(V_nodes, clause, count)
            all_dict = calculateFeature(V_nodes, "Horn-V")
            all_dict["instance"] = fname[2:]
            all_dict["Horn-fraction"] = count / nc
            d = d.append(all_dict, ignore_index=True)

d = d.set_index("instance")
d.to_csv("weighted_horn_feature.csv")
