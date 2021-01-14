#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 00:53:23 2020

@author: chenzhiyi
"""

import os
import gzip
from pysat.formula import WCNF 
import json
import numpy as np
import pandas
from pandas.io.json import json_normalize
class Node:
    def __init__(self):
        self.edges = set()
        
def calculateFeature(nodes, prefix):
    all_degree = []
    result_dic = {}
    for n in nodes: 
        all_degree.append(len(n.edges))
    result_dic[prefix + "-" + 'mean'] = np.mean(all_degree)
    result_dic[prefix + "-" + 'max'] = np.max(all_degree)
    result_dic[prefix + "-" + 'min'] = np.min(all_degree)
    result_dic[prefix + "-" + 'std'] = np.std(all_degree)
    return result_dic

def addVGEdge(nodes, clause):
    clause = list(map(lambda c: abs(c), clause))
    for n in clause:
        nodes[n - 1].edges.update(clause)
        
def addVCGEdge(V_nodes, C_node, clause, count_C):
    clause = list(map(lambda c: abs(c), clause))
    for n in clause:
        V_nodes[n - 1].edges.add(count_C)
    C_node.edges.update(clause)
        
common = ['mean', 'max', 'min', 'std']
VG = list(map(lambda s: 'VG-' + s, common))
VCG = list(map(lambda s: 'VCG-' + s, common))
all_feature = ['instance'] + VG + VCG

d = pandas.DataFrame(data=[], columns=all_feature)

os.chdir("/Users/chenzhiyi/Desktop/HonoursProgram/maxsat_instances/ms_evals/MS19/mse19-incomplete-unweighted-benchmarks")
for root, dirs, files in os.walk("."):
    for file in files:
        if (file.endswith(".gz")):
            VG_nodes = []
            V_nodes = []
            C_nodes = []
            fname = os.path.join(root, file)
            print(fname)
            f = gzip.open(fname, 'rt')
            wcnf = WCNF(from_fp=f)
            for n in range(wcnf.nv):
                VG_nodes.append(Node())
                V_nodes.append(Node())
                
            count_C = 1
            for c in wcnf.hard:
                addVGEdge(VG_nodes, c)
                
                C_node = Node()
                addVCGEdge(V_nodes, C_node, c, count_C)
                C_nodes.append(C_node)
                count_C += 1
            for c in wcnf.soft:
                addVGEdge(VG_nodes, c)
                
                C_node = Node()
                addVCGEdge(V_nodes, C_node, c, count_C)
                C_nodes.append(C_node)
                count_C += 1
                
            all_dict = calculateFeature(VG_nodes, "VG")
            all_dict.update(calculateFeature(V_nodes + C_nodes, "VCG"))
            all_dict["instance"] = fname[2:]
            d = d.append(all_dict, ignore_index=True)
d = d.set_index('instance')
d.to_csv("feature_graph.csv")
            