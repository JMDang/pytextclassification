#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   base_classfier.py
Author:   dangjinming(jmdang777@qq.com)
Desc  :   base_classfier
"""

# logging.basicConfig(format='%(asctime)s-%(levelname)s - %(message)s', level=logging.INFO)

class ClassifierBase(object):
    """
    Abstract Base Class
    """

    def train(self, train_data, eval_data, model_dir: str, **kwargs):
        raise NotImplementedError('train method not implemented.')

    def predict(self, predict_data: list):
        raise NotImplementedError('predict method not implemented.')

    def eval(self, **kwargs):
        raise NotImplementedError('eval method not implemented.')

    def load_model(self):
        raise NotImplementedError('load method not implemented.')

    def save_model(self):
        raise NotImplementedError('save method not implemented.')
