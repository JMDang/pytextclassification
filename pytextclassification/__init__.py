#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  author:dangjinming(jmdang777@qq.com)
  Desc  : 
"""

import sys

__version__ = '0.0.1'

from pytextclassification.rnn_classfier import TextRNNClassifier
from pytextclassification.cnn_classfier import TextCNNClassifier
from pytextclassification.fasttext_classifier import FastTextClassifier
from pytextclassification.classical_classifier import ClassicClassifier
from pytextclassification.transformers_classification import *