# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

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
            data.append((t['text'][0], self.label_0))
        return data

    def predict(self, texts):
        test_data = self.process(texts)
        test_data_ = self._get_data(test_data, self.label_id, set_type='predict')

        intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data_)

        result = []
        for t, r in zip(texts, intent_preds_list):
            result.append({"id": t['id'], "label": r})

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
    texts = [{"id": "0", "text": ["其实我愿意吞声忍气，但却怕被看不起。"]},
             {"id": "1", "text": ["跟自己说对不起，总是为难自己，让自己生活在水深火热中，对不起。我要买漂亮的衣服来犒劳自己。"]}]
    lcp = LanguageModelClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)

    predict_file = '../data_predict/ocemotion_predict.json'
    data = read_test_data(predict_file)
    res = lcp.predict(data)

    output_file = '../submit/ocemotion_predict.json'
    f = open(output_file, 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    f.close()
