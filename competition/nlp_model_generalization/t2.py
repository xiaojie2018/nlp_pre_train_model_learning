#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 11:27
# @Author  : xiaojie
# @Site    : 
# @File    : t2.py
# @Software: PyCharm


import os
import json
import random
import pandas as pd


file_path = "E:\\bishai\\数据集\\NLP中文预训练模型泛化能力挑战赛\\"


def read_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    data_num = []
    for line in lines:
        line = line.replace('\n', '').split('\t')
        data.append(line)
        data_num.append(len(line))
    data_num = sorted(list(set(data_num)))
    print(file)
    print(data_num)
    print('*'*60)
    res = []
    for d in data:
        res.append({'id': d[0], "text": d[1:]})
    return res


def wee(file, outfile):
    data = read_data(file)
    f = open(outfile, 'w', encoding='utf-8')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    f.close()


if __name__ == '__main__':

    flags = ['OCEMOTION', "OCNLI", "TNEWS"]
    for flag in flags:
        file = file_path + '{}_a.csv'.format(flag)
        wee(file, './data_predict/{}_predict.json'.format(flag.lower()))

