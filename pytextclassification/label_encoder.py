#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   label_encoder.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   label_encoder
"""
import sys
import os

class LabelEncoder(object):
    """类别处理工具
    """
    def __init__(self, label_dict_or_path):
        """初始化类别编码类
        [in]  label_dict_or_path: str/dict, 类别及其对应id的信息
        """
        if isinstance(label_dict_or_path, str) and os.path.exists(label_dict_or_path):
            self.label_to_id = LabelEncoder.load_label_vocab(label_dict_or_path)
        elif isinstance(label_dict_or_path, dict):
            self.label_to_id = label_dict_or_path
        else:
            raise ValueError("unknown label_dict_or_path type: {}".format(type(label_dict_or_path)))

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def transform(self, label_name):
        """类别名称转id
        [in]  label_name: str, 类别名称
        [out] label_id: id, 类别名称对应的id
        """
        if label_name not in self.label_to_id:
            raise ValueError("unknown label name: %s" % label_name)
        return self.label_to_id[label_name]

    def inverse_transform(self, label_id):
        """id转类别名称
        [in]  label_id: id, 类别名称对应的id
        [out] label_name: str, 类别名称
        """
        if label_id not in self.id_to_label :
            raise ValueError("unknown label id: %s" % label_id)
        return self.id_to_label [label_id]

    def size(self):
        """返回类别数
        [out] label_num: int, 类别树目
        """
        return len(set(self.label_to_id.keys()))

    def labels(self):
        """返回全部类别 并固定顺序
        """
        return sorted(set(self.label_to_id.keys()))

    @staticmethod
    def load_label_vocab(label_path):
        """load_vocabulary
        """
        label_to_id = {}
        with open(label_path, "r") as fr:
            for line in fr:
                parts = line.strip("\n").split("\t")
                label_to_id[parts[1]] = int(parts[0])
        return label_to_id

    def save_label_vocabulary(self, filepath):
        """
        save_label_vocabulary
        """
        with open(filepath, "w") as fw:
            for idx in range(len(self.id_to_label)):
                fw.write(str(idx) + "\t" + self.id_to_label[idx] + "\n")

if __name__ == "__main__":
    label_encoder = LabelEncoder("../input/label.txt")
    print(label_encoder.label_to_id)