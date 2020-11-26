# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
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
        self.label_id = {int(k): v for k, v in self.config.label_id.items()}
        self.label_0 = self.config.labels[0]
        self.trainer = Trainer(self.config)
        self.trainer.load_model()

    def process(self, texts):
        return texts
        data = []
        for t in texts:
            data.append((t, self.label_0))
        return data

    def predict(self, texts):
        test_data = self.process(texts)
        test_data_ = self._get_data(test_data, self.label_id, set_type='predict')

        intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data_)

        result = []
        for t, r in zip(texts, intent_preds_list):
            result.append([t['id1'], t['id2'], r])

        return result

        # result = []
        # for s in intent_preds_list_all:
        #     s1 = {}
        #     for k, v in s.items():
        #         s1[k] = round(v, 6)
        #     result.append(s1)
        #
        # # return result
        # return [[x, y] for x, y in zip(texts, intent_preds_list)]


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # data.append(line.replace('\n', ''))
            data.append(eval(line))
    return data


if __name__ == '__main__':

    file = "./output/model_ernie_1112_2"
    texts = [{"id1": "0", "id2": "0", "text1": "东区西区？什么时候下证？", "text2": "我在给你发套", "label": 0},
             {"id1": "0", "id2": "1", "text1": "东区西区？什么时候下证？", "text2": "您看下我发的这几套", "label": 0}]
    lcp = LanguageModelClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)

    predict_file = 'D:\\bishai\\data\\ccf2020\\房产行业聊天问答匹配\\test\\test.json'
    data = read_test_data(predict_file)
    res = lcp.predict(data)

    o_file = './tijiao/text11_12_2.tsv'
    f = open(o_file, 'w', encoding='utf-8')
    for r in res:
        r1 = '{}\t{}\t{}\n'.format(str(r[0]), str(r[1]), str(r[2]))
        f.write(r1)
    f.close()
