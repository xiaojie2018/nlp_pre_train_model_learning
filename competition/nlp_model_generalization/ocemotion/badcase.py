#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 11:25
# @Author  : xiaojie
# @Site    : 
# @File    : badcase.py
# @Software: PyCharm

import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
# print(project_path)
sys.path.append(project_path)
import json
from ocemotion.utils import ClassificationDataPreprocess
from argparse import Namespace
from ocemotion.trainer import Trainer
import numpy as np
import openpyxl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LanguageModelClassificationPredict(ClassificationDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'classification_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelClassificationPredict, self).__init__(self.config)
        self.label_id = self.config.label_id
        self.label_0 = self.config.labels[0]
        self.trainer = Trainer(self.config)
        self.trainer.load_model()

    def process(self, texts):
        data = []
        for t in texts:
            # data.append({"text": t['text'][0], "label": self.label_0})
            data.append((t['text'], t['label']))
        return data

    def predict(self, texts):
        test_data = self.process(texts)
        test_data_ = self._get_data(test_data, self.label_id, set_type='predict')

        intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data_)

        result = []
        for t, r in zip(texts, intent_preds_list):
            result.append({"text": t['text'], "true": t['label'], "predict": r})

        return result


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # data.append(line.replace('\n', ''))
            data.append(eval(line))
    return data


if __name__ == '__main__':
    file = "./output/model_ernie_1126_1"
    texts = [{"text": "现在对我来说最恶心的事就是把嘴上破掉的皮一不小心咽肚子里去了......", "label": "sadness"},
             {"text": "苦逼的加班......[泪流满面]", "label": "sadness"}]
    lcp = LanguageModelClassificationPredict(file)
    print(lcp.config.labels)
    res = lcp.predict(texts)
    print(res)

    flag = ['train', 'test'][0]

    predict_file = '../data/ocemotion_{}.json'.format(flag)
    data = read_test_data(predict_file)
    res = lcp.predict(data)

    # 判断badcase
    labels = lcp.config.labels
    error_m = np.zeros((len(labels), len(labels)), dtype=int)
    error = []
    for r in res:
        if r['true'] != r['predict']:
            id1 = labels.index(r['true'])
            id2 = labels.index(r['predict'])
            error_m[id1][id2] += 1
            error.append(r)

    output_file = '../badcase/ocemotion_{}_error.json'.format(flag)
    f = open(output_file, 'w', encoding='utf-8')
    for d in error:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    f.close()

    otf = '../badcase/ocemotion_{}_error_maxtix.xlsx'.format(flag)
    excel = openpyxl.Workbook(otf)
    sheet = excel.create_sheet(0)
    sheet.append([""]+labels)
    for ind, s in enumerate(error_m.tolist()):
        r = [labels[ind]] + s
        sheet.append(r)
    excel.save(otf)

