#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   self_layers.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   自定义layer
"""

import torch
import torch.nn as nn
INF = 1.0 * 1e12

class LAST_STEP(nn.Module):
    def __init__(self):
        super(LAST_STEP, self).__init__()
    def forward(self, inputs):
        x = torch.squeeze(inputs[:,0:1,:], dim=1)
        return x

class WORD_ATT_V1(nn.Module):
    def __init__(self,fea_size, attention_size):
        super(WORD_ATT_V1, self).__init__()
        self.attention_size = attention_size
        self.fea_size = fea_size
        # Word-level attention network
        self.word_att = nn.Linear(self.fea_size, self.attention_size )#全连接相当于之前矩阵乘法操作
        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(self.attention_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, inputs, mask=None):
        att_score = self.word_att(inputs) #b,s,attention_size
        att_score = self.tanh(att_score) #b,s,attention_size
        att_score = self.word_context_vector(att_score)# b,s,1
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = mask.float()
            # mask = mask.type(torch.float32)
            mask = mask.unsqueeze(dim = -1)
            inf_tensor = torch.full(size=mask.shape, dtype=torch.float32, fill_value=-INF)
            att_score = torch.multiply(att_score, mask) + torch.multiply(inf_tensor, (1 - mask))

        att_weight = self.softmax(att_score)
        x_weighted = torch.sum(torch.multiply(inputs, att_weight), dim=1)
        return x_weighted, att_weight

class WORD_ATT_V2(nn.Module):
    def __init__(self,fea_size):
        super(WORD_ATT_V2, self).__init__()
        self.fea_size = fea_size
        self.word_att = nn.Linear(self.fea_size, 1)
        self.query = self.create_parameter(shape=[self.fea_size, 1], dtype="float32")
        self.tanh2 = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, mask=None):
        att_score = self.word_att(inputs)
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = mask.float()
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = torch.full(size=mask.shape, dtype=torch.float32, fill_value=-INF)
            att_score = torch.multiply(att_score, mask) + torch.multiply(inf_tensor, (1 - mask))
        att_weight = self.softmax(att_score)
        x_weighted = torch.sum(torch.multiply(inputs, att_weight), dim=1)
        return x_weighted, att_weight

class SelfAttention(nn.Module):
    """
    A close implementation of attention network of ACL 2016 paper,
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification (Zhou et al., 2016).
    ref: https://www.aclweb.org/anthology/P16-2034/
    Args:
        hidden_size (int): The number of expected features in the input x.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.att_weight = nn.parameter.Parameter(torch.ones(1, hidden_size, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask=None):
        """
        Args:
            input (paddle.Tensor) of shape (batch, seq_len, input_size): Tensor containing the features of the input sequence.
            mask (paddle.Tensor) of shape (batch, seq_len) :
                Tensor is a bool tensor, whose each element identifies whether the input word id is pad token or not.
                Defaults to `None`.
        """
        forward_input, backward_input = torch.chunk(input, chunks=2, dim=2)
        # elementwise-sum forward_x and backward_x
        # Shape: (batch_size, max_seq_len, hidden_size)
        h = torch.add([forward_input, backward_input])
        # Shape: (batch_size, hidden_size, 1)
        att_weight = self.att_weight.tile((h.size(0), 1, 1))
        # Shape: (batch_size, max_seq_len, 1)
        att_score = torch.bmm(torch.tanh(h), att_weight)
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = mask.float()
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = torch.full(size=mask.shape, dtype=torch.float32, fill_value=-INF)
            att_score = torch.multiply(att_score, mask) + torch.multiply(inf_tensor, (1 - mask))
            # Shape: (batch_size, max_seq_len, 1)
        att_weight = self.softmax(att_score)
        # Shape: (batch_size, lstm_hidden_size)
        reps = torch.bmm(torch.transpose(h, 1, 2), att_weight).squeeze(-1)
        reps = torch.tanh(reps)
        return reps, att_weight

class SelfInteractiveAttention(nn.Module):
    """
    A close implementation of attention network of NAACL 2016 paper, Hierarchical Attention Networks for Document Classiﬁcation (Yang et al., 2016).
    ref: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Args:
        hidden_size (int): The number of expected features in the input x.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.input_weight = self.create_parameter(shape=[1, hidden_size, hidden_size], dtype="float32")
        self.bias = self.create_parameter(shape=[1, 1, hidden_size], dtype="float32")
        self.att_context_vector = self.create_parameter(shape=[1, hidden_size, 1], dtype="float32")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask=None):
        """
        Args:
            input (paddle.Tensor) of shape (batch, seq_len, input_size): Tensor containing the features of the input sequence.
            mask (paddle.Tensor) of shape (batch, seq_len) :
                Tensor is a bool tensor, whose each element identifies whether the input word id is pad token or not.
                Defaults to `None
        """
        weight = self.input_weight.tile(repeat_times=(input.size(0), 1, 1))#BHH
        bias = self.bias.tile(repeat_times=(input.size(0), 1, 1))
        # Shape: (batch_size, max_seq_len, hidden_size)
        word_squish = torch.bmm(input, weight) + bias


        att_context_vector = self.att_context_vector.tile(repeat_times=(input.shape[0], 1, 1))
        # Shape: (batch_size, max_seq_len, 1)
        att_score = torch.bmm(word_squish, att_context_vector)
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = mask.float()
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = torch.full(size=mask.shze(), dtype=torch.float32, fill_value=-INF)
            att_score = torch.multiply(att_score, mask) + torch.multiply(inf_tensor, (1 - mask))
        att_weight = self.softmax(att_score, axis=1)

        # Shape: (batch_size, hidden_size)
        reps = torch.bmm(input.transpose(perm=(0, 2, 1)), att_weight).squeeze(-1)
        return reps, att_weight