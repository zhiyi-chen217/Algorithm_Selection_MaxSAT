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
from scipy.stats import entropy

def entropyHelper(data, base=None):
    value,counts = np.unique(data, return_counts=True)
    return entropy(counts, base=base)

def calculateBalance(data, prefix):
    all_ratio = []
    result_dic = {}
    for f in data: 
        f = np.array(f)
        all_ratio.append(np.sum(f < 0)/len(f))
    result_dic[prefix + "-" + 'mean'] = np.mean(all_ratio)
    result_dic[prefix + "-" + 'max'] = np.max(all_ratio)
    result_dic[prefix + "-" + 'min'] = np.min(all_ratio)
    result_dic[prefix + "-" + 'std'] = np.std(all_ratio)
    result_dic[prefix + "-" + 'entropy'] = entropyHelper(all_ratio)
    return result_dic
        
balance = ['balance-hard', 'balance-soft', 'balance-hard-mean', 'balance-hard-max', 'balance-hard-min', 'balance-hard-std', 'balance-hard-entropy',
           'balance-soft-mean', 'balance-soft-max', 'balance-soft-min', 'balance-soft-std', 'balance-soft-entropy']
hard_clause = ['nhard_len_stats.ave', 'nhard_len_stats.max',
       'nhard_len_stats.min', 'nhard_len_stats.stddev', 'nhards']
soft_clause = ['nsoft_len_stats.ave', 'nsoft_len_stats.max', 'nsoft_len_stats.min',
       'nsoft_len_stats.stddev', 'nsoft_wts', 'nsofts']
soft_weight = ['soft_wt_stats.ave', 'soft_wt_stats.max', 'soft_wt_stats.min',
       'soft_wt_stats.stddev']
whole_formula = [ 'instance', 'ncls', 'nvars']

all_feature = balance + hard_clause + soft_clause + soft_weight + whole_formula

d = pandas.DataFrame(data=[], columns=all_feature)

os.chdir("/Users/chenzhiyi/Desktop/HonoursProgram/maxsat_instances/ms_evals/MS19/mse19-incomplete-unweighted-benchmarks")

for root, dirs, files in os.walk("."):
    for file in files:
        if (file.endswith(".gz")):
            fname = os.path.join(root, file)
            print(fname)
            f = gzip.open(fname, 'rt')
            wcnf = WCNF(from_fp=f)
            add = False
            comment = ""
            for line in wcnf.comments:
                if line.startswith("c-") :
                    break
                if add:
                    comment = comment + line[1:] + "\n"
                elif line.startswith("c{") :
                    comment = comment + line[1:] + "\n"
                    add = True
            j = json.loads(comment)
            if len(wcnf.hard) > 0 :
                flat_hard = np.concatenate(wcnf.hard).flat
                bhard = np.sum(flat_hard < 0) / len(flat_hard)
                balance_hard_dic = calculateBalance(wcnf.hard, 'balance-hard')
            else :
                bhard = 0
                balance_hard_dic = {}
            if len(wcnf.soft) > 0 :
                flat_soft = np.concatenate(wcnf.soft).flat
                bsoft = np.sum(flat_soft < 0) / len(flat_soft)
                balance_soft_dic = calculateBalance(wcnf.soft, 'balance-soft')
            else :
                bsoft = 0
                balance_soft_dic = {}
            j["balance-hard"] = bhard
            j["balance-soft"] = bsoft
            j["instance"] = fname[2:]
            j.update(balance_hard_dic)
            j.update(balance_soft_dic)
            j = json_normalize(j)
            d = d.append(j, ignore_index=True)

d = d.set_index('instance')
d.to_csv("feature_balance.csv")
            