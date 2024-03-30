#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   text_cnn_classfier.py
Author:   dangjinming(jmdang777@qq.com)
Desc  :   text_cnn_classfier
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, Adafactor

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    roc_curve,
    auc,
    average_precision_score,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from sklearn.model_selection import train_test_split

sys.path.append('..')
from pytextclassification.helper import set_seed, multi_classify_prf_macro, multi_classify_prf_micro, flatten_results
from pytextclassification.utils import load_data
from pytextclassification.vocab import Vocab
from pytextclassification.label_encoder import LabelEncoder
from pytextclassification.tokenizer import CommonTokenizer
from pytextclassification.base_classifier import ClassifierBase
from pytextclassification.layers import SelfAttention
from pytextclassification.config.model_args import ClassificationArgs

cur_path = os.path.abspath(os.path.dirname(__file__))


class SelfDataset(Dataset):
    """self dataset"""

    def __init__(self, datas):
        ids, bigram, trigram, labels, lengths= zip(*datas)
        self.ids = np.array(ids)
        self.bigram = np.array(bigram)
        self.trigram = np.array(trigram)
        self.labels = np.array(labels)
        self.lengths = np.array(lengths)
        

    def __getitem__(self, index):
        return self.ids[index], self.bigram[index], self.trigram[index], self.labels[index], self.lengths[index], 

    def __len__(self):
        return len(self.ids)

def biGramHash(input_ids, index, n_gram_buckets_size):
            t1 = input_ids[index - 1] if index - 1 >= 0 else 0
            return (t1 * 7777777) % n_gram_buckets_size

def triGramHash(input_ids, index, n_gram_buckets_size):
    t1 = input_ids[index - 1] if index - 1 >= 0 else 0
    t2 = input_ids[index - 2] if index - 2 >= 0 else 0
    return (t2 * 8888888 *  + t1 * 7777777) % n_gram_buckets_size

def build_dataset(
    X, 
    y,
    multi_label=False,
    labels_sep=",",
    vocab_level="word",
    word_vocab_path=None,
    label_vocab_path=None,
    max_seq_length=64,
    max_vocab_size=100000,
    enable_ngram=True,
    n_gram_buckets_size=200000
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
        labels = set()
        for label in y.tolist():
            labels.update(label.split(labels_sep))
        label_id_map = {la: i for i, la in enumerate(labels)}
        label_encoder = LabelEncoder(label_id_map)
        label_encoder.save_label_vocabulary(label_vocab_path)

    train_datasets = []
    for content, label in zip(X, y):
        if not content:
            continue
        input_ids = tokenizer.encode(content)
        seq_len = len(input_ids)
        if len(input_ids) < max_seq_length:
            input_ids.extend([vocab[vocab.pad_token]] * (max_seq_length - len(input_ids)))
        else:
            input_ids = input_ids[:max_seq_length]
            seq_len = max_seq_length
        # fasttext ngram
        bigram = []
        trigram = []
        if enable_ngram:
            # ------ngram------
            for i in range(max_seq_length):
                bigram.append(biGramHash(input_ids, i, n_gram_buckets_size))
                trigram.append(triGramHash(input_ids, i, n_gram_buckets_size))
        else:
            bigram = [0] * max_seq_length
            trigram = [0] * max_seq_length

        if not multi_label:
            target_label = label_encoder.transform(label)
        else:
            target_label = [0.0] * label_encoder.size()
            for ln in label.split(labels_sep):
                target_label[label_encoder.transform(ln)] = 1
        train_datasets.append((input_ids, bigram, trigram, target_label, seq_len))
    return SelfDataset(train_datasets), vocab, label_encoder


class FastTextModel(nn.Module):
    """Bag of Tricks for Efficient Text Classification"""

    def __init__(
            self,
            vocab_size,
            num_classes,
            embed_size=300,
            n_gram_buckets_size=700000,
            fast_text_hidden_size=256,
            fast_text_dropout_rate=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedding_bigram = nn.Embedding(n_gram_buckets_size, embed_size)
        self.embedding_trigram= nn.Embedding(n_gram_buckets_size, embed_size)
        self.dropout = nn.Dropout(fast_text_dropout_rate)
        self.fc1 = nn.Linear(embed_size * 3, fast_text_hidden_size)
        self.fc2 = nn.Linear(fast_text_hidden_size, num_classes)

    def forward(self, inputs, bigram, trigram, true_lengths=None):
        word_embeded = self.embedding(inputs)
        bigram_embeded = self.embedding_bigram(bigram)
        trigram_embeded = self.embedding_trigram(trigram)
        embeded_cat = torch.cat((word_embeded, bigram_embeded, trigram_embeded), -1)

        out = embeded_cat.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class FastTextClassifier(ClassifierBase):
    """FastTextClassifier"""

    def __init__(
            self,            
            multi_label=False,
            enable_ngram=True,
            n_gram_buckets_size=700000,
            fast_text_hidden_size=256,
            fast_text_dropout_rate=0.5,
            use_cuda=True,
            cuda_device=-1,
            args=None,
            labels_sep=",",
            **kwargs
    ):
        """
        Init the FastTextClassifier
        @param tast_text_dropout_rate:
        """
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError("'use_cuda' set to True when cuda is unavailable."
                                 " Make sure CUDA is available or set use_cuda=False.")
        logger.info(f'device: {self.device}')
        
        self.args =  ClassificationArgs()
        if isinstance(args, dict):
            self.args.update_from_dict(args)
            assert self.args.vocab_level in ["char", "word"], "vocab_level must be in  [char, word]"
        elif isinstance(args, ClassificationArgs):
            self.args = args
        self.args.update_from_dict(
            {
                "multi_label": multi_label,
                "use_cuda": use_cuda,
                "labels_sep": labels_sep,  # if multi_label is true,it will be valid
                "n_gram_buckets_size": n_gram_buckets_size,
                "fast_text_hidden_size": fast_text_hidden_size,
                "fast_text_dropout_rate": fast_text_dropout_rate
            }
        )
        self.model = None

    def __str__(self):
        return f'FastTextClassifier instance ({self.model})'

    def train(
            self,
            train_data,
            output_dir=None,
            show_running_loss=True,
            args=None,
            eval_data=None,
            **kwargs,
    ):
        """
        Train model with train_data and save model to output_dir
        @param train_data:
        @param output_dir:
        @param show_running_loss:
        @param args:
        @param eval_data:
        @return:
        """
        SEED = 1024
        set_seed(SEED)

        logger.info('train model...')
        if args:
            self.args.updata_from_dict(args)
        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            logger.warning("evaluate_during_training is enabled but eval_data is not specified."
                           " Pass eval_data to model.train() if using evaluate_during_training."
                           "now disabling evaluate_during_training.")
            self.args.evaluate_during_training = False
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

        # 1.加载数据
        X_train, y_train = load_data(train_data)
        # 2.初始化dataset实例，并得到vocab和label_encoder 实例(如果是测试集合，实例来自于保存好的vocabulary文件)
        """如果word_vocab_path和label_vocab_path存在，则加载，不存在则从训练集构建"""
        train_dataset, vocab, label_encoder = build_dataset(
            X_train, 
            y_train,
            multi_label=self.args.multi_label,
            labels_sep = self.args.labels_sep,
            vocab_level = self.args.vocab_level,
            word_vocab_path = word_vocab_path,
            label_vocab_path = label_vocab_path,
            max_seq_length = self.args.max_seq_length,
            max_vocab_size = self.args.max_vocab_size,
            enable_ngram=self.args.enable_ngram,
            n_gram_buckets_size=self.args.n_gram_buckets_size,

        )

        train_iter = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        dev_iter = None
        if eval_data:
            X_eval, y_eval = load_data(eval_data)
            eval_dataset, _, _ = build_dataset(
                X_eval, y_eval,
                multi_label=self.args.multi_label,
                labels_sep = self.args.labels_sep,
                vocab_level = self.args.vocab_level,
                word_vocab_path = word_vocab_path,
                label_vocab_path = label_vocab_path,
                max_seq_length = self.args.max_seq_length,
                max_vocab_size = self.args.max_vocab_size,
                enable_ngram=self.args.enable_ngram,
                n_gram_buckets_size=self.args.n_gram_buckets_size,
            )

            dev_iter = DataLoader(eval_dataset, batch_size=self.args.batch_size, shuffle=False)
        vocab_size = len(vocab)
        num_labels = label_encoder.size()
        self.args.update_from_dict(
                {
                    "num_labels": num_labels
                }
        )
        self.args.labels_map = label_encoder.label_to_id
        self.args.labels_list = sorted(label_encoder.label_to_id.keys())
        logger.info(f'vocab_size:{vocab_size}, num_labels:{num_labels}, labels_map:{self.args.labels_map}')
        logger.info(f'train_data_size:{len(train_dataset)}, dev_data_size: {len(dev_iter) if dev_iter else "no dev_data"}')

        # 3. 创建model
        self.model = FastTextModel(
            vocab_size=vocab_size,
            num_classes=num_labels,
            embed_size=self.args.embed_size,
            n_gram_buckets_size=self.args.n_gram_buckets_size,
            fast_text_hidden_size=self.args.fast_text_hidden_size,
            fast_text_dropout_rate=self.args.fast_text_dropout_rate
        )
        self._move_model_to_device()
        # 4. 训练循环
        # train model
        global_step, training_details = self.train_cycle(
            train_iter,
            dev_iter,
            **kwargs
        )
        logger.info('train model done')
        return global_step, training_details

    def train_cycle(
            self,
            train_iter,
            dev_iter=None,
            show_running_loss=True,
            verbose=True,
            **kwargs,
    ):
        history = []
        # train
        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_dir)

        start_time = time.time()

        if not self.args.multi_label:
            criterion = nn.CrossEntropyLoss()
        else:
            # criterion = nn.BCELoss()
            criterion = nn.BCEWithLogitsLoss()  # 要求输入logits是未经过sigmoid的
        optimizer_grouped_parameters = []
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
            )
        elif self.args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adafactor_eps,
                clip_threshold=self.args.adafactor_clip_threshold,
                decay_rate=self.args.adafactor_decay_rate,
                beta1=self.args.adafactor_beta1,
                weight_decay=self.args.weight_decay,
                scale_parameter=self.args.adafactor_scale_parameter,
                relative_step=self.args.adafactor_relative_step,
                warmup_init=self.args.adafactor_warmup_init,
            )
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    self.args.optimizer
                )
            )

        global_step = 0
        training_progress_scores = collections.defaultdict(list)
        best_eval_metric = None
        early_stopping_counter = 0
        tr_loss, logging_loss = 0.0, 0.0

        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.args.num_train_epochs))
            for step, (input_ids, bigram, trigram, labels, lengths) in enumerate(train_iter):
                input_ids = input_ids.to(self.device)
                bigram = bigram.to(self.device)
                trigram = trigram.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, bigram, trigram)
                loss = criterion(outputs, labels)

                if show_running_loss:
                    logger.info(f"Running Epoch {epoch+1}/{self.args.num_train_epochs}. Running Loss: {loss:9.4f}")

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / self.args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            self.args.output_dir, "checkpoint-{}".format(global_step)
                        )
                        self.save_model(
                            output_dir_current, optimizer, model=self.model
                        )
                    if self.args.evaluate_during_training and (
                            self.args.evaluate_during_training_steps > 0
                            and global_step % self.args.evaluate_during_training_steps == 0
                    ):
                        metric_results, _ = self.eval(
                            dev_iter,
                            verbose=True,
                            **kwargs
                        )
                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(loss.item())
                        for key in metric_results:
                            training_progress_scores[key].append(metric_results[key])

                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                self.args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )
                        tb_writer.flush()

                        if not best_eval_metric:
                            best_eval_metric = metric_results[self.args.early_stopping_metric]
                            self.save_model(
                                self.args.best_model_dir,
                                optimizer,
                                model=self.model,
                                results=metric_results
                            )
                        if best_eval_metric and self.args.early_stopping_metric_minimize:
                            if (
                                    best_eval_metric - metric_results[self.args.early_stopping_metric]
                                    > self.args.early_stopping_delta
                            ):
                                best_eval_metric = metric_results[self.args.early_stopping_metric]
                                self.save_model(
                                    self.args.best_model_dir,
                                    optimizer,
                                    model=self.model,
                                    results=metric_results,
                                )
                                early_stopping_counter = 0
                            else:
                                if self.args.use_early_stopping:
                                    if (
                                            early_stopping_counter
                                            < self.args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {self.args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(
                                                f" Early stopping patience: {self.args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {self.args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                        return (
                                            global_step,
                                            loss
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                    metric_results[self.args.early_stopping_metric] - best_eval_metric
                                    > self.args.early_stopping_delta
                            ):
                                best_eval_metric = metric_results[self.args.early_stopping_metric]
                                self.save_model(
                                    self.args.best_model_dir,
                                    optimizer,
                                    model=self.model,
                                    results=metric_results,
                                )
                                early_stopping_counter = 0
                            else:
                                if self.args.use_early_stopping:
                                    if (
                                            early_stopping_counter
                                            < self.args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {self.args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {self.args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {self.args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                        return (
                                            global_step,
                                            loss
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        self.model.train()

            output_dir_current = os.path.join(
                self.args.output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch+1)
            )
            if self.args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, model=self.model)
        return (
            global_step,
            loss
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval(
            self,
            dev_iter,
            output_dir=None,
            verbose=True,
            **kwargs,
    ):
        """
        @param eval_data:
        @param output_dir:
        @param verbose:
        @param kwargs:
        """
        if not output_dir:
            output_dir = self.args.output_dir
        self._move_model_to_device()
        return self.eval_cycle(
            dev_iter,
            output_dir,
            verbose=verbose,
            **kwargs,
        )

    def eval_cycle(
            self,
            data_iter,
            output_dir,
            verbose=True,
            **kwargs
    ):
        """
        Eval model.
        @param data_iter:
        @return: accuracy score, loss
        """

        if not self.args.multi_label:
            criterion = nn.CrossEntropyLoss()
        else:
            # criterion = nn.BCELoss()
            criterion = nn.BCEWithLogitsLoss()  # 要求输入logits是未经过sigmoid的

        if not self.model:
            raise ValueError('model not trained.')
        self.model.eval()
        logger.info(" eval start}")
        loss_total = 0.0
        eval_steps = 0
        n_batches = len(data_iter)
        eval_data_nums = sum([len(input_ids) for (input_ids, bigram, trigram, labels, lengths) in data_iter])
        pred_labels = np.empty((eval_data_nums, self.args.num_labels))
        if self.args.multi_label:
            true_labels = np.empty((eval_data_nums, self.args.num_labels))
        else:
            true_labels = np.empty((eval_data_nums))

        with torch.no_grad():
            for i, (input_ids, bigram, trigram, labels, lengths) in enumerate(data_iter):
                input_ids = input_ids.to(self.device)
                bigram = bigram.to(self.device)
                trigram = trigram.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, bigram, trigram)
                loss = criterion(outputs, labels)
                if self.args.multi_label:
                    outputs = outputs.sigmoid()

                loss_total += loss.item()
                eval_steps += 1

                start_index = self.args.batch_size * i
                end_index = (
                    start_index + self.args.batch_size
                    if i != (n_batches - 1)
                    else eval_data_nums
                )
                pred_labels[start_index:end_index] = outputs.detach().cpu().numpy()
                true_labels[start_index:end_index] = labels.detach().cpu().numpy()

            eval_loss = loss_total / eval_steps
            # if multi_label is True, it is the model output after sigmoid processing,else it is logits
            model_outputs = pred_labels 
            if not self.args.multi_label:
                pred_labels = np.argmax(pred_labels, axis=1)

            result = self.compute_metrics(
                pred_labels, model_outputs, true_labels, **kwargs
            )

            result["eval_loss"] = eval_loss

            logger.info(f" eval done. eval datas num {eval_data_nums}, eval res: {result}")

        return result, model_outputs

    def predict(
        self, 
        to_predict
    ):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """
        assert isinstance(to_predict, list), "input must be a list"
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = np.empty((len(to_predict), self.args.num_labels))
        if self.args.multi_label:
            out_label_ids = np.empty((len(to_predict), self.args.num_labels))
        else:
            out_label_ids = np.empty((len(to_predict)))
        out_label_names = []

        word_vocab_file = os.path.join(self.args.output_dir, 'word_vocab.txt')
        label_vocab_file = os.path.join(self.args.output_dir, 'label_vocab.txt')
        if not os.path.exists(word_vocab_file) \
            or not os.path.exists(label_vocab_file):
            logger.warning(f"can't find file {word_vocab_file} or {label_vocab_file}, maybe you should train model first.")
            raise IOError('File not found, maybe model is not trained')
        
        vocab = Vocab(word_vocab_file)
        label_encoder = LabelEncoder(label_vocab_file)
        tokenizer = CommonTokenizer(vocab)

        to_predict_ids = []
        to_predict_bigram = []
        to_predict_trigram = []
        for content in to_predict:
            input_ids = tokenizer.encode(content)
            seq_len = len(input_ids)

            if len(input_ids) < self.args.max_seq_length:
                input_ids.extend([vocab[vocab.pad_token]] * (self.args.max_seq_length - len(input_ids)))
            else:
                input_ids = input_ids[:self.args.max_seq_length]
                seq_len = self.args.max_seq_length

            # fasttext ngram
            bigram = []
            trigram = []
            if self.args.enable_ngram:
                # ------ngram------
                for i in range(self.args.max_seq_length):
                    bigram.append(biGramHash(input_ids, i, self.args.n_gram_buckets_size))
                    trigram.append(triGramHash(input_ids, i,self.args.n_gram_buckets_size))
            else:
                bigram = [0] * max_seq_length
                trigram = [0] * max_seq_length
            to_predict_ids.append(input_ids)
            to_predict_bigram.append(bigram)
            to_predict_trigram.append(trigram)

        if self.args.multi_label:
            out_label_ids = np.empty((len(to_predict), self.args.num_labels))
        else:
            out_label_ids = np.empty((len(to_predict)))

        if to_predict_ids:
            n_batches = len(to_predict_ids) // self.args.batch_size  \
                if len(to_predict_ids) % self.args.batch_size == 0 \
                else len(to_predict_ids) // self.args.batch_size + 1
            for i in range(n_batches):
                start_index = self.args.batch_size * i
                end_index = (
                    start_index + self.args.batch_size
                    if i != (n_batches - 1)
                    else len(to_predict_ids)
                )
                batch_input_ids = to_predict_ids[start_index:end_index]
                batch_bigram = to_predict_bigram[start_index:end_index]
                batch_trigram = to_predict_trigram[start_index:end_index]
                batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
                batch_bigram = torch.tensor(batch_bigram, dtype=torch.long).to(self.device)
                batch_trigram = torch.tensor(batch_trigram, dtype=torch.long).to(self.device)

                logits = self.model(batch_input_ids, batch_bigram, batch_trigram)
                if self.args.multi_label:
                    threshold_values = self.args.threshold if self.args.threshold else 0.5
                    pre_scores = logits.detach().cpu().sigmoid().numpy()
                    pred_labels = [
                        [self._threshold(pred, threshold_values) for pred in example]
                        for example in pre_scores
                    ]
                    
                    pred_labels_names = [[label_encoder.inverse_transform(index) for index, prob_ in enumerate(example) if prob_] for example in pred_labels]
                    pred_labels_names = [self.args.labels_sep.join(example) for example in pred_labels_names]
                    
                else:
                    pre_scores = torch.softmax(logits.detach().cpu(), dim=1).numpy()
                    pred_labels = np.argmax(pre_scores, axis=1)
                    pred_labels_names = [label_encoder.inverse_transform(lid)for lid in pred_labels]

                out_label_ids[start_index:end_index] = pred_labels
                out_label_names.extend(pred_labels_names)
        assert len(to_predict) == len(out_label_ids), f"len of predict {len(to_predict)} != len of out_label_ids"
        return out_label_ids, out_label_names
    
    def load_model(self):
        """
        Load model from output_dir
        @return:
        """

        model_path = os.path.join(self.args.best_model_dir, 'pytorch_model.bin')
        word_vocab_file = os.path.join(self.args.output_dir, 'word_vocab.txt')
        label_vocab_file = os.path.join(self.args.output_dir, 'label_vocab.txt')
        if not os.path.exists(word_vocab_file) \
            or not os.path.exists(label_vocab_file) \
            or not os.path.exists(model_path):
            logger.warning(f"can't find file {word_vocab_file} or {label_vocab_file} or {model_path}, maybe you should train model first.")
            raise IOError('File not found, maybe model is not trained')
            
        vocab = Vocab(word_vocab_file)
        label_encoder = LabelEncoder(label_vocab_file)

        vocab_size = len(vocab)
        num_labels = label_encoder.size()

        #创建model
        self.model = FastTextModel(
            vocab_size=vocab_size,
            num_classes=num_labels,
            embed_size=self.args.embed_size,
            n_gram_buckets_size=self.args.n_gram_buckets_size,
            fast_text_hidden_size=self.args.fast_text_hidden_size,
            fast_text_dropout_rate=self.args.fast_text_dropout_rate
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self._move_model_to_device()


    def save_model(
        self, 
        output_dir=None, 
        optimizer=None, 
        model=None,
        results=None
    ):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and self.args.save_optimizer:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def compute_metrics(
            self,
            preds,
            model_outputs,
            labels,
            **kwargs,
    ):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            model_outputs: Model outputs, if multi_label is True, it is the model output after sigmoid processing,else it is logits
            labels: Ground truth labels
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            For non-binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn).
            For binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn, AUROC, AUPRC).
        """ 
        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            if metric.startswith("prob_"):
                extra_metrics[metric] = func(labels, model_outputs)
            else:
                extra_metrics[metric] = func(labels, preds)

        if self.args.multi_label:
            threshold_values = self.args.threshold if self.args.threshold else 0.5

            pred_labels = [
                [self._threshold(pred, threshold_values) for pred in example]
                for example in preds
            ]
            mismatched = labels != pred_labels
        else:
            mismatched = labels != preds
            pred_labels = preds
        if self.args.multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            acc = accuracy_score(labels, np.array(pred_labels))
            precision, recall, f_score, true_sum = precision_recall_fscore_support(
                labels, np.array(pred_labels), average='weighted')
            return {**{
                "LRAP": label_ranking_score,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
            }, **extra_metrics}
        elif self.args.regression:
            return {**extra_metrics}

        # single label
        mcc = matthews_corrcoef(labels, preds)
        acc = accuracy_score(labels, preds)
        classification_report_str = classification_report(labels, preds, digits=4, output_dict=True)

        if self.args.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            precision, recall, f_score, true_sum = precision_recall_fscore_support(
                labels, preds, labels=[0, 1], average="binary")

            scores = np.array([softmax(element)[1] for element in model_outputs])
            fpr, tpr, thresholds = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
            auprc = average_precision_score(labels, scores)
            return (
                {
                    **{
                        "mcc": mcc,
                        "tp": tp,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "auroc": auroc,
                        "auprc": auprc,
                        "acc": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f_score,
                        "classification_report": classification_report_str,
                    },
                    **extra_metrics,
                }
            )
        else:
            precision, recall, f_score, true_sum = precision_recall_fscore_support(
                labels, preds, average='weighted')
            return {**{
                "mcc": mcc,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
                "classification_report": classification_report_str,
            }, **extra_metrics}


if __name__ == '__main__':
    
    train_args = ClassificationArgs()
    train_args.num_labels = 2
    train_args.num_train_epochs = 2
    train_args.batch_size = 128
    train_args.vocab_level = "char"
    train_args.enable_ngram = True

    fasttext = FastTextClassifier(multi_label=True, args=train_args)

    #模型训练
    fasttext.train(train_data="data/train.txt", eval_data="data/dev.txt")


    #加载在验证集效果最好的模型进行预测
    fasttext.load_model()
    preds, model_outputs = fasttext.predict([
        "柔弱的儿媳跪在地上向老头子苦苦哀求：“够了，适可而止！",
        "那里芳草丛生，他徘徊找不到入口，最终还是她领着他进来",
        "儿媳妇想来都是保守稳重的人，但是夜深人静后像变了个人似的",
        "终于找到了一个看书神器，全本小说免费看，关键还能赚钱！",
        "穷小子第一次下山就被退婚，谁知他有三重身份，战神，神医，龙王",
        "知与校花合租，穿越修真界的爷爷竟传我透视眼，我笑的合不拢嘴",
        ]
    )
    print(preds, model_outputs)

