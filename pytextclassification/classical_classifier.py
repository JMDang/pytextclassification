#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   classic_classfier.py
Author:   dangjinming(jmdang777@qq.com)
Desc  :   classic_classfier
"""

import collections
import argparse
import json
import os
import sys
import time
from scipy.special import softmax
from sklearn.metrics import precision_score, recall_score, f1_score
from loguru import logger

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pickle
from sklearn import metrics

sys.path.append('..')
from pytextclassification.helper import set_seed, multi_classify_prf_macro, multi_classify_prf_micro, flatten_results
from pytextclassification.utils import load_data, load_stop_words
from pytextclassification.vocab import Vocab
from pytextclassification.label_encoder import LabelEncoder
from pytextclassification.tokenizer import CommonTokenizer
from pytextclassification.base_classifier import ClassifierBase
from pytextclassification.layers import SelfAttention
from pytextclassification.config.model_args import ClassificationArgs

cur_path = os.path.abspath(os.path.dirname(__file__))

def build_dataset(
    X, 
    y,
    vocab_level="word",
    stopwords=[],
    word_vocab_path=None,
    label_vocab_path=None,
    max_vocab_size=1000000
):
    assert vocab_level in ["char", "word"], "vocab_level must be in  [char, word]"
    if os.path.exists(word_vocab_path):
        vocab = Vocab(word_vocab_path)
    else:
        vocab_dic = Vocab.build_vocab(X, vocab_level=vocab_level, max_size=max_vocab_size)
        vocab = Vocab(vocab_dic)
        vocab.save_vocabulary(word_vocab_path)
    tokenizer = CommonTokenizer(vocab)

    if os.path.exists(label_vocab_path):
        label_encoder = LabelEncoder(label_vocab_path)
    else:
        labels = set(y.tolist())
        label_id_map = {la: i for i, la in enumerate(labels)}
        label_encoder = LabelEncoder(label_id_map)
        label_encoder.save_label_vocabulary(label_vocab_path)

    train_datasets = []
    for content, label in zip(X, y):
        if not content:
            continue
        train_words = tokenizer.tokenize_with_stop_words(content, stopwords)
        if len(train_words) < 2:
            continue
        train_words = " ".join(train_words)
        target_label = label_encoder.transform(label)
        train_datasets.append((train_words, target_label))
    return train_datasets, vocab, label_encoder

class ClassicClassifier(ClassifierBase):
    """ClassicClassifier"""

    def __init__(
            self,            
            model_name_or_model='lr',
            feature_name_or_feature='tfidf',
            stopwords_path="./data/stopwords.txt",
            args=None,
            **kwargs
    ):
        """
        Init the ClassicClassifier
        @param cnn_dropout_rate:
        """
        
        self.args =  ClassificationArgs()
        if isinstance(args, dict):
            self.args.update_from_dict(args)
            assert self.args.vocab_level in ["char", "word"], "vocab_level must be in  [char, word]"
        elif isinstance(args, ClassificationArgs):
            self.args = args
        self.args.update_from_dict(
            {   
                "model_name_or_model": model_name_or_model,
                "feature_name_or_feature": feature_name_or_feature,
                "stopwords_path": stopwords_path
            }
        )
        if isinstance(model_name_or_model, str):
            model_name = model_name_or_model.lower()
            if model_name not in ['lr', 'random_forest', 'decision_tree', 'knn', 'bayes', 'xgboost', 'svm']:
                raise ValueError('model_name not found.')
            logger.debug(f'model_name: {model_name}')
            self.model = self.get_model(model_name)
        elif hasattr(model_name_or_model, 'fit'):
            self.model = model_name_or_model
        else:
            raise ValueError('model_name_or_model set error.')

        if isinstance(feature_name_or_feature, str):
            feature_name = feature_name_or_feature.lower()
            if feature_name not in ['tfidf', 'count']:
                raise ValueError('feature_name not found.')
            logger.debug(f'feature_name: {feature_name}')
            if feature_name == 'tfidf':
                self.feature = TfidfVectorizer(ngram_range=(1, 2), analyzer=self.args.vocab_level)
            else:
                self.feature = CountVectorizer(ngram_range=(1, 2), analyzer=self.args.vocab_level)
        elif hasattr(feature_name_or_feature, 'fit_transform'):
            self.feature = feature_name_or_feature
        else:
            raise ValueError('feature_name_or_feature set error.')


    def __str__(self):
        return f'ClassicClassifier instance ({self.model})'

    @staticmethod
    def get_model(model_type):
        if model_type in ["lr", "logistic_regression"]:
            model = LogisticRegression(solver='lbfgs', fit_intercept=False)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=300)
        elif model_type == "decision_tree":
            model = DecisionTreeClassifier()
        elif model_type == "knn":
            model = KNeighborsClassifier()
        elif model_type == "bayes":
            model = MultinomialNB(alpha=0.1, fit_prior=False)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError('xgboost not installed, please install it with "pip install xgboost"')
            model = XGBClassifier()
        elif model_type == "svm":
            model = SVC(kernel='linear', probability=True)
        else:
            raise ValueError('model type set error.')
        return model

    def train(
            self,
            train_data,
            output_dir=None,
            args=None,
            eval_data=None,
            **kwargs,
    ):
        """
        Train model with train_data and save model to output_dir
        @param train_data:
        @param output_dir:
        @param args:
        @param eval_data:
        @return:
        """
        SEED = 1024
        set_seed(SEED)

        logger.info('train model...')
        if args:
            self.args.updata_from_dict(args)

        if output_dir:
            self.args.update_from_dict(
                {
                    "output_dir": output_dir
                }
            )
        if (
                os.path.exists(self.args.output_dir)
                and os.listdir(self.args.output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Set overwrite_output_dir: True to automatically overwrite."
            )
        os.makedirs(self.args.output_dir, exist_ok=True)

        word_vocab_path = os.path.join(self.args.output_dir, 'word_vocab.txt')
        label_vocab_path = os.path.join(self.args.output_dir, 'label_vocab.txt')

        stopwords = load_stop_words(self.args.stopwords_path)
        # 1.加载数据
        X_train, y_train = load_data(train_data)
        # 2.初始化dataset实例，并得到vocab和label_encoder 实例(如果是测试集合，实例来自于保存好的vocabulary文件)
        """如果word_vocab_path和label_vocab_path存在，则加载，不存在则从训练集构建"""
        train_dataset, vocab, label_encoder = build_dataset(
            X_train, 
            y_train,
            vocab_level = self.args.vocab_level,
            stopwords = stopwords,
            word_vocab_path = word_vocab_path,
            label_vocab_path = label_vocab_path,
        )
        eval_dataset = None
        if eval_data:
            X_eval, y_eval = load_data(eval_data)
            eval_dataset, _, _ = build_dataset(
                X_eval, y_eval,
                vocab_level = self.args.vocab_level,
                stopwords = stopwords,
                word_vocab_path = word_vocab_path,
                label_vocab_path = label_vocab_path,
               
            ) 
        vocab_size = len(vocab)
        num_labels = label_encoder.size()
        self.args.labels_map = label_encoder.label_to_id
        self.args.labels_list = sorted(label_encoder.label_to_id.keys())
        logger.info(f'vocab_size:{vocab_size}, num_labels:{num_labels}, labels_map:{self.args.labels_map}')
        logger.info(f'train_data_size:{len(train_dataset)}, dev_data_size: {len(eval_dataset) if eval_dataset else "no dev_data"}')
        train_feat = self.feature.fit_transform([item[0] for item in train_dataset])
        # train model
        self.model.fit(train_feat, [item[1] for item in train_dataset])
        # evaluate
        eval_acc = self.eval(eval_dataset)
        # save model
        self.save_model()
        return eval_acc

    def eval(
            self,
            eval_dataset,
            output_dir=None,
            **kwargs,
    ):
        """
        @param eval_data:
        @param output_dir:
        @param kwargs:
        """
        if not output_dir:
            output_dir = self.args.output_dir
        pre_labels, pre_probs = self.predict([item[0] for item in eval_dataset])
        labels = [item[1] for item in eval_dataset]
        acc_score = metrics.accuracy_score(pre_labels, labels)
        return acc_score

    def predict(self, to_predict: list):
        """
        Performs predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
        """
        assert isinstance(to_predict, list), "input must be a list"
        stopwords = load_stop_words(self.args.stopwords_path)
        word_vocab_file = os.path.join(self.args.output_dir, 'word_vocab.txt')
        label_vocab_file = os.path.join(self.args.output_dir, 'label_vocab.txt')
        if not os.path.exists(word_vocab_file) \
            or not os.path.exists(label_vocab_file):
            logger.warning(f"can't find file {word_vocab_file} or {label_vocab_file}, maybe you should train model first.")
            raise IOError('File not found, maybe model is not trained')
        
        vocab = Vocab(word_vocab_file)
        label_encoder = LabelEncoder(label_vocab_file)
        tokenizer = CommonTokenizer(vocab)

        # tokenize text
        X_tokens = [" ".join(tokenizer.tokenize_with_stop_words(sen, stopwords)) for sen in to_predict] 
        # transform
        X_feat = self.feature.transform(X_tokens)
        predict_labels = self.model.predict(X_feat)
        probs = self.model.predict_proba(X_feat)
        predict_probs = [prob[np.where(self.model.classes_ == label)][0] for label, prob in zip(predict_labels, probs)]
        return predict_labels, predict_probs
    
    def load_pkl(self, pkl_path):
        """
        加载词典文件
        :param pkl_path:
        :return:
        """
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        return result

    def save_pkl(self, vocab, pkl_path, overwrite=True):
        """
        存储文件
        :param pkl_path:
        :param overwrite:
        :return:
        """
        if pkl_path and os.path.exists(pkl_path) and not overwrite:
            return
        if pkl_path:
            with open(pkl_path, 'wb') as f:
                pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)  # python3
    
    def load_model(self):
        """
        Load model from output_dir
        @return:
        """
        model_path = os.path.join(self.args.output_dir, 'classifier_model.pkl')
        if os.path.exists(model_path):
            self.model = self.load_pkl(model_path)
            feature_path = os.path.join(self.args.output_dir, 'classifier_feature.pkl')
            self.feature = self.load_pkl(feature_path)
            logger.info(f'Loaded model: {model_path}.')
        else:
            logger.error(f'{model_path} not exists.')


    def save_model(self):
        """
        Save model to output_dir
        @return:
        """
        feature_path = os.path.join(self.args.output_dir, 'classifier_feature.pkl')
        self.save_pkl(self.feature, feature_path)
        model_path = os.path.join(self.args.output_dir, 'classifier_model.pkl')
        self.save_pkl(self.model, model_path)
        logger.info(f'Saved model: {model_path}, feature_path: {feature_path}')

        return self.model, self.feature


    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)


if __name__ == '__main__':
    
    train_args = ClassificationArgs()
    train_args.vocab_level = "word"

    classic_classfier = ClassicClassifier(
        model_name_or_model='lr',
        feature_name_or_feature='tfidf',
        args=train_args)

    #模型训练
    classic_classfier.train(train_data="data/train.txt", eval_data="data/dev.txt")


    #加载在验证集效果最好的模型进行预测
    classic_classfier.load_model()
    preds, model_outputs = classic_classfier.predict([
        "柔弱的儿媳跪在地上向老头子苦苦哀求：“够了，适可而止！",
        "那里芳草丛生，他徘徊找不到入口，最终还是她领着他进来",
        "儿媳妇想来都是保守稳重的人，但是夜深人静后像变了个人似的",
        "终于找到了一个看书神器，全本小说免费看，关键还能赚钱！",
        "穷小子第一次下山就被退婚，谁知他有三重身份，战神，神医，龙王",
        "知与校花合租，穿越修真界的爷爷竟传我透视眼，我笑的合不拢嘴",
        ]
    )
    print(preds, model_outputs)

