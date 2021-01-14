#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:57:30 2020

@author: chenzhiyi
"""
import json
import gzip
import numpy as np
import pandas
from pysat.formula import WCNF 
from pandas.io.json import json_normalize
f = gzip.open('/Users/chenzhiyi/Desktop/HonoursProgram/maxsat_instances/ms_evals/MS19/mse19-incomplete-unweighted-benchmarks/aes/mul_8_13.wcnf.gz', 'rt')
wcnf = WCNF(from_fp=f)
comment = ""
add = False
for line in wcnf.comments:
    if line.startswith("c-") :
        break
    if add:
        comment = comment + line[1:] + "\n"
    elif line.startswith("c{") :
        comment = comment + line[1:] + "\n"
        add = True
j = json.loads(comment)
flat_hard = np.concatenate(wcnf.hard).flat
flat_soft = np.concatenate(wcnf.soft).flat
j["balance-hard"] =  np.sum(flat_hard < 0) / len(flat_hard)
j["balance-soft"] =  np.sum(flat_soft < 0) / len(flat_soft)
j = json_normalize(j)


d = pandas.DataFrame(data=[], columns=['balance-hard', 'balance-soft', 'ncls', 'nhard_len_stats.ave', 'nhard_len_stats.max',
       'nhard_len_stats.min', 'nhard_len_stats.stddev', 'nhards',
       'nsoft_len_stats.ave', 'nsoft_len_stats.max', 'nsoft_len_stats.min',
       'nsoft_len_stats.stddev', 'nsoft_wts', 'nsofts', 'nvars', 'sha1sum',
       'soft_wt_stats.ave', 'soft_wt_stats.max', 'soft_wt_stats.min',
       'soft_wt_stats.stddev'])
d = d.append(j)