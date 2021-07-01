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



all_feature = ['instance', 'problem']

d = pandas.DataFrame(data=[], columns=all_feature)


os.chdir("/Users/chenzhiyi/Desktop/HonoursProgram/maxsat_instances/ms_evals/MS19/mse19-incomplete-unweighted"
         "-benchmarks")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".gz"):
            fname = os.path.join(root, file)
            print(fname)
            fname = fname[2:]
            problem_name = fname[:fname.index('/')]
            all_dict = {"instance" : fname, "problem" : problem_name}
            d = d.append(all_dict, ignore_index=True)
d = d.set_index('instance')
d.to_csv("problem_label.csv")
