#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File  :   tokenizer.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   tokenizer
"""
import jieba

class BaseTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get_tokenizer(self):
        return self.tokenizer

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass


class CommonTokenizer(BaseTokenizer):
    """
    Constructs a tokenizer
    Args:
        vocab(pytextclassification.Vocab)
    """

    def __init__(self, vocab):
        super(CommonTokenizer, self).__init__(vocab)
        self.word_segmenter  = jieba.Tokenizer()
        # initialize tokenizer
        self.word_segmenter .FREQ = {key: 1 for key in self.vocab.token_to_idx.keys()}
        self.word_segmenter .total = len(self.word_segmenter .FREQ)
        self.word_segmenter .initialized = True

    def tokenize(self, sentence, cut_all=False, use_hmm=True):
        """
        The method used to cut the text to tokens.
        """
        if self.vocab.vocab_level == "word":
            return list(self.word_segmenter.lcut(sentence, cut_all, use_hmm))
        else:
            return [ch for ch in sentence]

    def encode(self, sentence, cut_all=False, use_hmm=True):
        """
        The method used to convert the text to ids. It will firstly call
        :meth:`to_tokens` method to cut the text to tokens. Then, convert tokens to
        ids using `vocab`.
        """
        words = self.tokenize(sentence, cut_all, use_hmm)
        return self.vocab.to_indices(words)