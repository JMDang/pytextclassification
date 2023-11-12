#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   base_classfier.py
Author:   dangjinming(jmdang777@qq.com)
Desc  :   base_classfier
"""

import logging
logging.basicConfig(format='%(asctime)s-%(levelname)s - %(message)s', level=logging.INFO)

import argparse
import json
import os
import sys
import time
from sklearn.metrics import  precision_score, recall_score, f1_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn import metrics
from sklearn.model_selection import train_test_split
sys.path.append('..')
from pytextclassification.helper import set_seed, multi_classify_prf_macro, multi_classify_prf_micro
from pytextclassification.utils import load_data
from pytextclassification.vocab import Vocab
from pytextclassification.label_encoder import LabelEncoder
from pytextclassification.tokenizer import CommonTokenizer
from pytextclassification.base_classifier import ClassifierBase
from pytextclassification.layers import SelfAttention


cur_path = os.path.abspath(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfDataset(Dataset):
    """self dataset"""
    def __init__(self, datas):
        ids, labels, lengths = zip(*datas)
        self.ids = np.array(ids)
        self.labels = np.array(labels)
        self.lengths = np.array(lengths)

    def __getitem__(self, index):
        return self.ids[index], self.labels[index], self.lengths[index]

    def __len__(self):
        return len(self.ids)

def build_dataset(X, y,
                  task_type="mc",
                  labels_sep=",",
                  vocab_level="char",
                  word_vocab_path=None,
                  label_vocab_path=None,
                  max_seq_length=10,
                  max_vocab_size=10000,
                  is_train=True):
    assert vocab_level in ["char", "word"], "vocab_level must be in  [char, word]"
    assert  is_train or (not is_train and os.path.exists(label_vocab_path)), \
        "if is_train is False, label_vocab_path must bot be Satisfied"
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
        labels = set()  # 统计label的种类
        for label in y.tolist():
            labels.update(label.split(labels_sep))
        label_id_map = {la:i for i, la in enumerate(labels)}
        label_encoder = LabelEncoder(label_id_map)

    contents = []
    for content, label in zip(X, y):
        if not content:
            continue
        input_ids = tokenizer.encode(content)
        seq_len = len(input_ids)
        if max_seq_length:
            if len(input_ids) < max_seq_length:
                input_ids.extend([vocab[vocab.pad_token]] * (max_seq_length - len(input_ids)))
            else:
                input_ids = input_ids[:max_seq_length]
                seq_len = max_seq_length
        if task_type == "mc":
            target_label = label_encoder.transform(label)
        else:
            target_label = [0] * label_encoder.size()
            for ln in label.split(labels_sep):
                target_label[label_encoder.transform(ln)] = 1
        contents.append((input_ids, target_label, seq_len))
    return SelfDataset(contents), vocab, label_encoder

class TextRNNAttModel(nn.Module):
    """Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification"""
    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_size=300,
        lstm_hidden_size=256,
        fc_hidden_size=128,
        lstm_layers=2,
        pooling_type="mean",
        dropout_rate = 0.3
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.word_emb = nn.Embedding(vocab_size, emb_size)

        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout_rate)
        self.att = SelfAttention(lstm_hidden_size)
        self.fc = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        self.tanch = nn.Tanh()
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, inputs, true_lengths=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.word_emb(inputs)
        encoded_text, _ = self.lstm(embedded_text,sequence_length=true_lengths)
        if self.pooling_type == 'sum':
            encoded_text_pool = torch.sum(encoded_text, dim=1)
        elif self.pooling_type == 'max':
            encoded_text_pool = torch.max(encoded_text, dim=1)
        elif self.pooling_type == 'mean':
            encoded_text_pool = torch.mean(encoded_text, dim=1)
        else:
            raise RuntimeError(
                "Unexpected pooling type %s ."
                "Pooling type must be one of sum, max and mean." %
                self.pooling_type)

        fc_out = self.tanch(self.fc(encoded_text_pool))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

class TextRNNClassifier(ClassifierBase):
    """TextRNNClassifier"""
    def __init__(
            self,
            output_dir="outputs",
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.5, batch_size=64, max_seq_length=128,
            embed_size=300, max_vocab_size=10000,
            vocab_level="char"

    ):
        """
        Init the TextRNNClassifier
        @param output_dir: 模型保存路径
        @param hidden_size: lstm隐藏层
        @param num_layers: lstm层数
        @param dropout_rate:
        @param batch_size:
        @param max_seq_length:
        @param embed_size:
        @param max_vocab_size:
        @param unk_token:
        @param pad_token:
        """
        self.output_dir = output_dir
        self.is_trained = False
        self.model = None
        logging.info(f'device: {device}')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.max_vocab_size = max_vocab_size
        self.vocab_level = vocab_level
        self.task_type = "mc" #
        assert self.task_type in ["mc", "ml"], \
            "vocab_level must be in  [mc, ml], which represent multi_class or multi_label"
        assert self.vocab_level in ["char", "word"], "vocab_level must be in  [char, word]"

    def __str__(self):
        return f'TextRNNClassifier instance ({self.model})'

    def train(
            self,
            data_list_or_path,
            output_dir: str,
            header=None,
            names=('labels', 'text'),
            delimiter='\t',
            test_size=0.1,
            num_epochs=20,
            labels_sep=",",
            learning_rate=1e-3,
            require_improvement=1000,
            evaluate_during_training_steps=100):
        """
        Train model with data_list_or_path and save model to output_dir
        @param data_list_or_path:
        @param header:
        @param names:
        @param test_size:
        @param num_epochs: epoch数
        @param learning_rate: 学习率
        @param require_improvement: 若超过1000batch效果还没提升，则提前结束训练
        @param evaluate_during_training_steps: 每隔多少step评估一次模型
        @return:
        """
        logging.info('train model...')
        output_dir = self.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        word_vocab_path = os.path.join(self.output_dir, 'word_vocab.txt')
        label_vocab_path = os.path.join(self.output_dir, 'label_vocab.txt')
        save_model_path = os.path.join(self.output_dir, 'model')

        SEED = 1024
        set_seed(SEED)
        # 1.加载数据，并自动判断是多分类还是多标签
        X, y, self.task_type = load_data(data_list_or_path, header=None,
                                  names=('labels', 'text'), delimiter='\t',
                                labels_sep=',', is_train=False)
        # 2.初始化dataset实例，并得到vocab和label_encoder 实例(如果是测试集合，实例来自于保存好的vocabulary文件)
        """如果word_vocab_path和label_vocab_path存在，则加载，不存在则从训练集构建"""
        dataset, vocab, label_encoder = build_dataset(X, y,
                      is_multi_class=is_multi_class,
                      labels_sep=labels_sep,
                      vocab_level=self.vocab_level,
                      word_vocab_path=word_vocab_path,
                      label_vocab_path=label_vocab_path,
                      max_seq_length=self.max_seq_length,
                      max_vocab_size=self.max_vocab_size,
                      is_train=True)
        # 3. 切分数据集为训练集和验证集
        test_size = int(len(ds) * test_size)
        train_size = len(ds) - test_size
        train_dataset, dev_dataset = random_split(dataset, [train_size, test_size])
        logging.info(f"train_data size: {len(train_dataset)}, dev_data size: {len(dev_dataset)}")
        logging.info(f'train_data sample:\n{train_dataset[:3]}\ndev_data sample:\n{dev_dataset[:3]}')

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_iter = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True)
        vocab_size = len(vocab)
        num_classes = label_encoder.size()
        logging.info(f'vocab_size:{vocab_size}', 'num_classes:', num_classes)
        # 4. 创建model
        self.model = TextRNNAttModel(
            vocab_size, num_classes,
            emb_size=self.embed_size,
            lstm_hidden_size=self.hidden_size,
            lstm_layers=self.num_layers,
            pooling_type="mean",
            dropout_rate=self.dropout_rate
        )
        self.model.to(device)
        # 5. 训练循环
        # train model
        history = self.training_steps(save_model_path,
                                      train_iter,
                                      dev_iter,
                                      num_epochs,
                                      learning_rate,
                                      require_improvement)
        self.is_trained = True
        logging.info('train model done')
        return history

    def training_steps(self, save_model_path, train_iter, dev_iter,
                       num_epochs=10, learning_rate=1e-3,
                       require_improvement=1000):
        history = []
        # train
        start_time = time.time()

        if self.task_type == "mc":
            criterion = nn.CrossEntropyLoss()
        else:
            # criterion = nn.BCELoss()
            criterion = nn.BCEWithLogitsLoss()  # 要求输入logits是未经过sigmoid的

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_batch = 0  # 记录进行到多少batch
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        self.model.train()
        for epoch in range(num_epochs):
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                outputs = self.model(trains)
                loss = criterion(outputs, labels)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if total_batch % evaluate_during_training_steps == 0:
                    # 输出在训练集和验证集上的效果
                    y_true = labels.cpu()
                    y_pred = torch.max(outputs, 1)[1].cpu()
                    计算prf
                    y_true = np.array(y_true).flatten()
                    y_pred = np.array(y_pred).flatten()
                    p, r, f = multi_classify_prf_macro(y_true, y_pred)
                    train_acc = metrics.accuracy_score(y_true, y_pred)
                    if dev_iter is not None:
                        dev_acc, dev_loss = self.evaluate(dev_iter)
                        if dev_loss < dev_best_loss:
                            dev_best_loss = dev_loss
                            torch.save(self.model.state_dict(), save_model_path)
                            logging.info(f'Saved model: {save_model_path}')
                            improve = '*'
                            last_improve = total_batch
                        else:
                            improve = ''
                        time_dif = time.time()- start_time
                        msg = 'Iter:{0:>6},Train Loss:{1:>5.2},Train Acc:{2:>6.2%},Val Loss:{3:>5.2},Val Acc:{4:>6.2%},Time:{5} {6}'.format(
                            total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)
                    else:
                        time_dif = time.time()- start_time
                        msg = 'Iter:{0:>6},Train Loss:{1:>5.2},Train Acc:{2:>6.2%},Time:{3}'.format(
                            total_batch, loss.item(), train_acc, time_dif)
                    logging.info(msg)
                    history.append(msg)
                    self.model.train()
                total_batch += 1
                if total_batch - last_improve > require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    logging.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        return history

    # def predict(self, sentences: list):
    #     """
    #     Predict labels and label probability for sentences.
    #     @param sentences: list, input text list, eg: [text1, text2, ...]
    #     @return: predict_label, predict_prob
    #     """
    #     if not self.is_trained:
    #         raise ValueError('model not trained.')
    #     if not self.model:
    #         self.load_model()
    #     self.model.eval()
    #     word_vocab_path = os.path.join(self.output_dir, 'word_vocab.txt')
    #     label_vocab_path = os.path.join(self.output_dir, 'label_vocab.txt')
    #     vocab = Vocab(word_vocab_path, vocab_level=self.vocab_level)
    #     label_encoder = LabelEncoder(label_vocab_path)
    #     tokenizer = CommonTokenizer(vocab)
    #     contents = []
    #     for sen in sentences:
    #         if not sen:
    #             continue
    #         input_ids = tokenizer.encode(sen)
    #         seq_len = len(input_ids)
    #         if self.max_seq_length:
    #             if len(input_ids) < self.max_seq_length:
    #                 input_ids.extend([vocab[vocab.pad_token]] * (self.max_seq_length - len(input_ids)))
    #             else:
    #                 input_ids = input_ids[:self.max_seq_length]
    #                 seq_len = self.max_seq_length
    #         contents.append((input_ids, 0, seq_len))
    #     predict_dataset = SelfDataset(contents)
    #     pre_iter = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=True)
    #
    #     # predict prob
    #     predict_all = []
    #     with torch.no_grad():
    #         for ids, _ in enumerate(pre_iter):
    #             logits = self.model(ids)
    #             if self.task_type == "mc":
    #                 pred = F.softmax(logits, dim=1).detach().cpu().numpy()
    #                 pred = np.argmax(pred, axis=1)
    #             else:
    #                 pred = F.sigmoid(logits).detach().cpu().numpy()
    #                 pred = [1 if l > 0.5 else 0 for l in pred]
    #             predict_all.extend(pred)
    #
    #     predict_probs = proba_all.tolist()
    #     self.model.train()
    #     return predict_labels, predict_probs
    #
    # def evaluate_model(self, data_list_or_path, header=None,
    #                    names=('labels', 'text'), delimiter='\t'):
    #     """
    #     Evaluate model.
    #     @param data_list_or_path:
    #     @param header:
    #     @param names:
    #     @param delimiter:
    #     @return:
    #     """
    #     X_test, y_test, df = load_data(data_list_or_path, header=header, names=names, delimiter=delimiter)
    #     self.load_model()
    #     data, word_id_map, label_id_map = build_dataset(
    #         self.tokenizer, X_test, y_test,
    #         self.word_vocab_path,
    #         self.label_vocab_path,
    #         max_vocab_size=self.max_vocab_size,
    #         max_seq_length=self.max_seq_length,
    #         unk_token=self.unk_token,
    #         pad_token=self.pad_token,
    #     )
    #     data_iter = build_iterator(data, device, self.batch_size)
    #     return self.evaluate(data_iter)[0]
    #
    # def evaluate(self, data_iter):
    #     """
    #     Evaluate model.
    #     @param data_iter:
    #     @return: accuracy score, loss
    #     """
    #     if not self.model:
    #         raise ValueError('model not trained.')
    #     self.model.eval()
    #     loss_total = 0.0
    #     predict_all = np.array([], dtype=int)
    #     labels_all = np.array([], dtype=int)
    #     with torch.no_grad():
    #         for texts, labels in data_iter:
    #             outputs = self.model(texts)
    #             loss = F.cross_entropy(outputs, labels)
    #             loss_total += loss
    #             labels = labels.cpu().numpy()
    #             predic = torch.max(outputs, 1)[1].cpu().numpy()
    #             labels_all = np.append(labels_all, labels)
    #             predict_all = np.append(predict_all, predic)
    #         logging.info(f"evaluate, last batch, y_true: {labels}, y_pred: {predic}")
    #     acc = metrics.accuracy_score(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter)
    #
    # def load_model(self):
    #     """
    #     Load model from output_dir
    #     @return:
    #     """
    #     model_path = os.path.join(self.output_dir, 'model.pth')
    #     if os.path.exists(model_path):
    #         word_vocab_path = os.path.join(self.output_dir, 'word_vocab.txt')
    #         label_vocab_path = os.path.join(self.output_dir, 'label_vocab.txt')
    #         save_model_path = os.path.join(self.output_dir, 'model')
    #         self.model = TextRNNAttModel(
    #             vocab_size, num_classes,
    #             emb_size=self.embed_size,
    #             lstm_hidden_size=self.hidden_size,
    #             lstm_layers=self.num_layers,
    #             pooling_type="mean",
    #             dropout_rate=self.dropout_rate
    #         )
    #         self.model.load_state_dict(torch.load(model_path, map_location=device))
    #         self.model.to(device)
    #         self.is_trained = True
    #     else:
    #         logging.error(f'{model_path} not exists.')
    #         self.is_trained = False
    #     return self.is_trained

if __name__ == '__main__':
    X, y, task_type = load_data("data/train", names=("text", "labels"),  is_train=True)
    print(X)
    ds, vocab, label_encoder = build_dataset(X, y, task_type=task_type, word_vocab_path = "./word_vocab.txt", label_vocab_path="./label_vocab.txt")

    train_size = int(len(ds) * 0.8)
    test_size = len(ds) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
    print(type(train_dataset))
    print(type(ds))
    for d in ds:
        print(d)

    # ds = SelfDataset([("x1", "y1"), ("x2", "y2"), ("x3", "y3")])
    dl = DataLoader(dataset=ds, batch_size=2)
    for ids, label, length in dl:
        print(ids)
        print(label)
        print(length)

    dl = DataLoader(dataset=train_dataset, batch_size=2)
    for ids, label, length in dl:
        print(ids)
        print(label)
        print(length)
        break