
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
        line = line.replace('\n', '').split('\t')[1:]
        data.append(line)
        data_num.append(len(line))
    data_num = sorted(list(set(data_num)))
    print(file)
    print(data_num)
    print('*'*60)
    res = []
    for d in data:
        res.append({'text': d[:-1], "label": d[-1]})
    return res


def data_train_test_split(data, rato=0.8):
    """
    :param data:[[text, label], []]
    :return:
    """
    res = {}
    for d in data:
        if d['label'] not in res:
            res[d['label']] = []
        res[d['label']].append(d)
    print({k: len(v) for k, v in res.items()})
    train_data = []
    test_data = []
    for k, v in res.items():
        train_len = int(rato*len(v))
        random.shuffle(v)
        if train_len <= 5:
            train_data += v
            test_data += v
        else:
            train_data += v[:train_len]
            test_data += v[train_len:]

    return train_data, test_data


def tongji_len(data_len):
    print(max(data_len))
    print(min(data_len))
    print(sum(data_len)/len(data_len))


def wee(file, data):
    f = open(file, 'w', encoding='utf-8')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    f.close()


def read_ocemotion_data(file, outfile):
    """
    :param file:
    :param outfile:
    309
    3
    47.49319213313162
    {'sadness': 11210, 'happiness': 7975, 'like': 3657, 'anger': 3657, 'fear': 525, 'surprise': 808, 'disgust': 3896}
    :return:
    """
    data = read_data(file)
    data_len = [len(d['text'][0]) for d in data]
    tongji_len(data_len)
    res = [{"text": d['text'][0], "label": d['label']} for d in data]
    train_data, test_data = data_train_test_split(res)
    train_file = outfile + '_train.json'
    test_file = outfile + '_test.json'
    wee(train_file, train_data)
    wee(test_file, test_data)


def read_ocnli_data(file, outfile):
    """
    :param file:
    :param outfile:
    107
    10
    35.576521204671174

    50
    7
    24.165969427206218

    60
    2
    11.410551777464956
    {'0': 16779, '1': 17182, '2': 16476}
    :return:
    """
    data = read_data(file)
    data_len = [len(''.join(d['text'])) for d in data]
    data_len1 = [len(d['text'][0]) for d in data]
    data_len2 = [len(d['text'][1]) for d in data]
    tongji_len(data_len)
    tongji_len(data_len1)
    tongji_len(data_len2)
    res = [{"text": d['text'][0], "text_b": d['text'][1], "label": d['label']} for d in data]
    train_data, test_data = data_train_test_split(res)
    train_file = outfile + '_train.json'
    test_file = outfile + '_test.json'
    wee(train_file, train_data)
    wee(test_file, test_data)


def read_tnews_data(file, outfile):
    """
    :param file:
    :return:
    """
    """
    145
    2
    22.15952023988006
    {'108': 3437, '104': 5200, '106': 2107, '112': 3368, '109': 5955, '103': 3991, '116': 3390, '101': 4081, 
     '107': 4118, '100': 1111, '102': 4976, '110': 3632, '115': 2886, '113': 4851, '114': 257}
    """
    data = read_data(file)
    data_len = [len(d['text'][0]) for d in data]
    tongji_len(data_len)
    res = [{"text": d['text'][0], "label": d['label']} for d in data]
    train_data, test_data = data_train_test_split(res)
    train_file = outfile + '_train.json'
    test_file = outfile + '_test.json'
    wee(train_file, train_data)
    wee(test_file, test_data)


if __name__ == '__main__':
    flags = [('OCEMOTION', read_ocemotion_data), ("OCNLI", read_ocnli_data), ("TNEWS", read_tnews_data)]
    for flag, f in flags:
        file = file_path + '{}_train.csv'.format(flag)
        data = f(file, './data/{}'.format(flag.lower()))
