#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File  :   vocab.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   vocab
"""
import collections
import io
import json
import numpy as np
import os
import warnings
import jieba
from tqdm import tqdm

class Vocab(object):
    """The class used to convert between tokens and ids. It also includes some
    store/load functions.
    """
    def __init__(self, vocab_dict_or_path, unk_token="[UNK]", pad_token="[PAD]"):
        """vocab init
        """
        assert vocab_dict_or_path, "token_to_idx should not be None"
        self.unk_token = unk_token
        self.pad_token = pad_token

        if isinstance(vocab_dict_or_path, dict):
            self.vocab_level = self._get_vocab_level(vocab_dict_or_path)
            if unk_token not in vocab_dict_or_path and pad_token not in vocab_dict_or_path:
                for key in vocab_dict_or_path:
                    vocab_dict_or_path[key] += 2
            vocab_dict_or_path[pad_token] = 0
            vocab_dict_or_path[unk_token] = 1

            self._token_to_idx = vocab_dict_or_path
            self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        elif isinstance(vocab_dict_or_path, str) and os.path.exists(vocab_dict_or_path):
            self._token_to_idx = Vocab.load_vocabulary(vocab_dict_or_path,
                                      unk_token=self.unk_token,
                                      pad_token=self.pad_token)
            self.vocab_level = self._get_vocab_level( self._token_to_idx)
            self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        else:
            raise TypeError("unknown vocab_dict_or_path type: {}".format(type(vocab_dict_or_path)))

    def to_tokens(self, indices):
        """
        Maps the input indices to token list.
        """
        if not isinstance(indices, (list, tuple, np.ndarray)):
            indices = [indices]
        if isinstance(indices, (list, tuple)):
            indices = np.asarray(indices)
        if isinstance(indices, (np.ndarray)) and len(indices.shape) > 1:
            raise ValueError(
                "Token indices is invalid. Expected 1D array, but received {}D array. ".format(len(indices.shape))
            )
        tokens = []
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                warnings.warn(
                    "The type of `to_tokens()`'s input `indices` is not `int` which will be forcibly transfered to `int`. "
                )
                idx = int(idx)

            try:
                tokens.append(self.idx_to_token[idx])
            except KeyError:
                raise ValueError("Token index {} in the provided `indices` is invalid.".format(idx))

        return tokens

    def to_indices(self, tokens):
        """
        Maps the input tokens into indices.
        """
        return self[tokens]

    def _get_vocab_level(self, vocab_dic):
        """"""
        num_greter_than_1 = 0
        for key in vocab_dic:
            if len(key) > 1:
                num_greter_than_1 += 1
        if num_greter_than_1 > 9:
            return "word"
        return "char"

    def __getitem__(self, tokens):
        """tokens：list/tuple/str"""
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, self._token_to_idx[self.unk_token])
        else:
            return [
                self._token_to_idx.get(token, self._token_to_idx[self.unk_token]) for token in tokens
            ]

    def __len__(self):
        return len(self.token_to_idx)

    def __call__(self, tokens):
        return self[tokens]

    @property
    def idx_to_token(self):
        # Returns index-token dict
        return self._idx_to_token

    @property
    def token_to_idx(self):
        # Return token-index dict
        return self._token_to_idx

    @staticmethod
    def load_vocabulary(vocab_path, unk_token="[UNK]", pad_token="[PAD]"):
        """load_vocabulary
        """
        token_to_idx = {}
        token_to_idx[pad_token] = 0
        token_to_idx[unk_token] = 1
        with open(vocab_path, "r", encoding="utf-8") as fr:
            for index, line in enumerate(fr):
                token = line.rstrip("\n")
                token_to_idx[token] = token_to_idx.get(token, len(token_to_idx))
        return token_to_idx

    def save_vocabulary(self, filepath):
        """
        save_vocabulary
        """
        with open(filepath, "w", encoding='utf-8') as f:
            for idx in range(len(self.idx_to_token)):
                f.write(self.idx_to_token[idx] + "\n")

    @staticmethod
    def build_vocab(texts, max_size=100000, min_freq=1, unk_token="[UNK]", pad_token="[PAD]", vocab_level="word"):
        assert vocab_level in ["char", "word"], "vocab_level must be in  [char, word]"
        word_segmenter = jieba.Tokenizer()
        vocab_dic = {}
        for line in tqdm(texts):
            line = line.strip()
            if not line:
                continue
            content = line.split('\t')[0]
            if vocab_level == "char":
                for word in content:
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
            else:
                for word in word_segmenter.lcut(content, False, True):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: index+2 for index, word_count in enumerate(vocab_list)}
        vocab_dic.update({pad_token: 0, unk_token: 1})
        return vocab_dic
    
    def get_unk_token_id(self):
        return self._token_to_idx[self.unk_token] if self.unk_token is not None else self.unk_token

    def get_pad_token_id(self):
        return self._token_to_idx[self.pad_token] if self.pad_token is not None else self.pad_token

if __name__ == "__main__":
    # vocab = Vocab("./vocab")
    # print(vocab.token_to_idx)
    wd = Vocab.build_vocab(["你来自哪里"], max_size=100, min_freq=1, vocab_level="word")
    print(wd)
    vocab = Vocab(wd)
    print(len(vocab))
    print(vocab.to_indices("来自"))
    print(vocab[["来自", "哪里"]])


