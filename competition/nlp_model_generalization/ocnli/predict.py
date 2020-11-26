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
from ocnli.utils import SimilarityDataPreprocess
from argparse import Namespace
from ocnli.trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LanguageModelClassificationPredict(SimilarityDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'classification_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelClassificationPredict, self).__init__(self.config)
        self.label_id = {k: v for k, v in self.config.label_id.items()}
        self.label_0 = self.config.labels[0]
        self.trainer = Trainer(self.config)
        self.trainer.load_model()

    def process(self, texts):
        data = []
        for t in texts:
            data.append({"id": t['id'], "text": t['text'][0], "text_b": t['text'][1], "label": self.label_0})
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
    texts = [{"id": "0", "text": ["来回一趟象我们两个人要两千五百块美金.", "我们有急事需要来回往返"]},
             {"id": "1", "text": ["这个就被这门功课给卡下来了.", "这门功课挂科了"]}]
    lcp = LanguageModelClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)

    predict_file = '../data_predict/ocnli_predict.json'
    data = read_test_data(predict_file)
    res = lcp.predict(data)

    output_file = '../output/ocnli_predict.json'
    f = open(output_file, 'w', encoding='utf-8')
    for d in res:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    f.close()
