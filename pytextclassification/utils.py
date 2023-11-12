#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File  :   utils.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   utils
"""

from collections import defaultdict

import numpy as np
import os
import re
import pandas as pd
import logging

logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)


def load_data(data_list_or_path, header=None, names=('labels', 'text'), delimiter='\t',
              labels_sep=',', is_train=False):
    """
    Encoding data_list text
    @param data_list_or_path: list of (label, text), eg: [(label, text), (label, text) ...]
    @param header: read_csv header
    @param names: read_csv names
    @param delimiter: read_csv sep
    @param labels_sep: multi label split
    @param is_train: is train data
    @return: X, y, data_df
    """
    if isinstance(data_list_or_path, list):
        data_df = pd.DataFrame(data_list_or_path, columns=names)
    elif isinstance(data_list_or_path, str) and os.path.exists(data_list_or_path):
        data_df = pd.read_csv(data_list_or_path, header=header, delimiter=delimiter, names=names)
    elif isinstance(data_list_or_path, pd.DataFrame):
        data_df = data_list_or_path
    else:
        raise TypeError('should be list or file path, eg: [(label, text), ... ]')

    task_type = "mc"
    X, y = data_df['text'], data_df['labels']
    labels = set()#统计label的种类
    if y.size:
        for label in y.tolist():
            label_split = label.split(labels_sep)
            labels.update(label_split)
            if len(label_split) > 1:
                task_type = "ml"
        num_classes = len(labels)
        labels = sorted(list(labels))
        logging.info(f'loaded data list, X size: {len(X)}, y size: {len(y)}')
        if is_train:
            logging.info('num_classes: %d, labels: %s' % (num_classes, labels))
    assert len(X) == len(y)
    assert task_type in ["mc", "ml"], \
        "vocab_level must be in  [mc, ml], which represent multi_class or multi_label"
    return X, y, task_type

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'

def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)

def is_number(uchar):
    """判断一个unicode是否是数字"""
    return '\u0030' <= uchar <= '\u0039'

def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    return '\u0041' <= uchar <= '\u005a' or '\u0061' <= uchar <= '\u007a'

def is_alphabet_string(string):
    """判断是否全部为英文字母"""
    return all(is_alphabet(c) for c in string)

def is_alphabet_number_string(string):
    """判断全是数字和英文字符"""
    return all((is_alphabet(c) or is_number(c)) for c in string)

def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    return not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar))

def remove_punctuation(strs):
    """
    去除标点符号
    :param strs:
    :return:
    """
    return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())

if __name__ == '__main__':
    load_data()