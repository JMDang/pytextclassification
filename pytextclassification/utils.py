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


def load_data(data, header=None, names=('labels', 'text'), delimiter='\t',
              labels_sep=','):
    """
    Encoding data_list text
    @param data: list of (label, text),  or pd.DataFrame or str
    @param header: read_csv header
    @param names: read_csv names
    @param delimiter: read_csv sep
    @param labels_sep: multi label split
    @param is_train: is train data
    @return: X, y, data_df
    """
    if isinstance(data, list):
        data_df = pd.DataFrame(data, columns=names)
    elif isinstance(data, str) and os.path.exists(data):
        data_df = pd.read_csv(data, header=header, delimiter=delimiter, names=names)
    elif isinstance(data, pd.DataFrame):
        data_df = data
    else:
        raise TypeError('should be list or file path, and file path must be exist, list eg: [(label, text), ... ]')

    X, y = data_df['text'], data_df['labels']
    labels = set()#统计label的种类
    if y.size:
        for label in y.tolist():
            label_split = label.split(labels_sep)
            labels.update(label_split)
        num_classes = len(labels)
        labels = sorted(list(labels))
        logging.info(f'loaded data list, X size: {len(X)}, y size: {len(y)}')
    assert len(X) == len(y)
    return X, y

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

def load_stop_words(stop_word_file):
    """
    加载停用词表
    :param stop_word_file: 停用词表文件路径
    :return: list of stop words
    """
    if not os.path.exists(stop_word_file):
        logging.warning("stop word file doesn't exist")
        return []
    with open(stop_word_file, "r", encoding="utf-8") as fr:
        lines = [line.strip() for line in fr]
    return lines


if __name__ == '__main__':
    load_data()