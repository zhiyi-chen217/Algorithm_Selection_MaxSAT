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
from scipy.stats import entropy

class Node:
    def __init__(self):
        self.edges = set()

def entropyHelper(data, base=None):
    value,counts = np.unique(data, return_counts=True)
    return entropy(counts, base=base)


def calculateFeature(nodes, prefix):
    all_degree = []
    result_dic = {}
    for n in nodes:
        all_degree.append(len(n.edges))
    result_dic[prefix + "-" + 'mean'] = np.mean(all_degree)
    result_dic[prefix + "-" + 'max'] = np.max(all_degree)
    result_dic[prefix + "-" + 'min'] = np.min(all_degree)
    result_dic[prefix + "-" + 'std'] = np.std(all_degree)
    result_dic[prefix + "-" + 'entropy'] = entropyHelper(all_degree)
    return result_dic


def addVGEdge(nodes, clause):
    clause = list(map(lambda c: abs(int(c)), clause))
    for n in clause:
        nodes[n - 1].edges.update(clause)


def addVCGEdge(V_nodes, C_node, clause, count_C):
    clause = list(map(lambda c: abs(int(c)), clause))
    for n in clause:
        V_nodes[n - 1].edges.add(count_C)
    C_node.edges.update(clause)


common = ['mean', 'max', 'min', 'std', 'entropy']
VG = list(map(lambda s: 'VG-' + s, common))
VCG = list(map(lambda s: 'VCG-' + s, common))
all_feature = ['instance'] + VG + VCG

d = pandas.DataFrame(data=[], columns=all_feature)


os.chdir("/Users/chenzhiyi/Desktop/HonoursProgram/maxsat_unweighted_300_2018/maxsat_instances/ms_evals/MS18/mse18-incomplete-unweighted-benchmarks/")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".gz"):
            VG_nodes = []
            V_nodes = []
            C_nodes = []
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
            
            for n in range(nv):
                VG_nodes.append(Node())
                V_nodes.append(Node())
            for n in range(nc):
                C_nodes.append(Node())

            for i in range(nc):
                s = f.readline()
                clause = s.split(" ")
                clause = clause[1:len(clause) - 1]
                addVGEdge(VG_nodes, clause)
                addVCGEdge(V_nodes, C_nodes[i], clause, i + 1)
            all_dict = calculateFeature(VG_nodes, "VG")
            all_dict.update(calculateFeature(V_nodes + C_nodes, "VCG"))
            all_dict["instance"] = fname[2:]
            d = d.append(all_dict, ignore_index=True)
d = d.set_index('instance')
d.to_csv("feature_graph.csv")
