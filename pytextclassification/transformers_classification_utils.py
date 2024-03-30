# -*- coding: utf-8 -*-
"""
author:dangjinming(jmdang777@qq.com)
"""

import json
import linecache
import os
import sys
import warnings

try:
    from collections import Iterable, Mapping
except ImportError:
    from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from io import open
from multiprocessing import Pool
from multiprocessing import cpu_count
from numbers import Real
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from loguru import logger
from datasets import Dataset as HFDataset
from datasets import load_dataset


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
            self, guid, text_a, text_b=None, label=None
    ):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(
            {
                "guid": self.guid,
                "text_a": self.text_a,
                "text_b": self.text_b,
                "label": self.label,
            }
        )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def preprocess_data_multiprocessing(data):
    text_a, text_b, tokenizer, max_seq_length = data

    examples = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return examples


def preprocess_batch_for_hf_dataset(dataset, tokenizer, max_seq_length):
    if "text_b" in dataset:
        return tokenizer(
            text=dataset["text_a"],
            text_pair=dataset["text_b"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )
    else:
        return tokenizer(
            text=dataset["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )


def preprocess_data(text_a, text_b, labels, tokenizer, max_seq_length):
    return tokenizer(
        text=text_a,
        text_pair=text_b,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )


def build_classification_dataset(
        data, tokenizer, args, mode, multi_label, output_mode, no_cache
):
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_{}_{}_{}_{}_{}".format(
            mode,
            args.model_type,
            args.max_seq_length,
            len(args.labels_list),
            len(data),
        ),
    )

    if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
    ):
        data = torch.load(cached_features_file)
        logger.info(f" Features loaded from cache at {cached_features_file}")
        examples, labels = data
    else:
        logger.debug(" Converting to features started. Cache is not used.")

        if len(data) == 3:
            # Sentence pair task
            text_a, text_b, labels = data
        else:
            text_a, labels = data
            text_b = None

        # If labels_map is defined, then labels need to be replaced with ints or floats
        if args.labels_map and not args.regression:
            if multi_label:
                if isinstance(labels[0], str):
                    labels = [[float(1) if i in label.split(args.labels_sep) else float(0) for i in
                               args.labels_list] for label in labels]
                elif isinstance(labels[0], list) and isinstance(labels[0][0], float):
                    labels = labels
                else:
                    labels = [[float(args.labels_map[l]) for l in label] for label in labels]
                # labels for multi_label need to be float
                print("djm165", labels)
            else:
                labels = [args.labels_map[label] for label in labels]

        if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
        ):
            if args.multiprocessing_chunksize == -1:
                chunksize = max(len(data) // (args.process_count * 2), 500)
            else:
                chunksize = args.multiprocessing_chunksize

            if text_b is not None:
                data = [
                    (
                        text_a[i: i + chunksize],
                        text_b[i: i + chunksize],
                        tokenizer,
                        args.max_seq_length,
                    )
                    for i in range(0, len(text_a), chunksize)
                ]
            else:
                data = [
                    (text_a[i: i + chunksize], None, tokenizer, args.max_seq_length)
                    for i in range(0, len(text_a), chunksize)
                ]

            with Pool(args.process_count) as p:
                examples = list(
                    tqdm(
                        p.imap(preprocess_data_multiprocessing, data),
                        total=len(text_a),
                        disable=args.silent,
                    )
                )

            examples = {
                key: torch.cat([example[key] for example in examples])
                for key in examples[0]
            }
        else:
            examples = preprocess_data(
                text_a, text_b, labels, tokenizer, args.max_seq_length
            )

        if output_mode == "classification":
            if args.multi_label:
                labels = torch.tensor(labels, dtype=torch.float)
            else:
                labels = torch.tensor(labels, dtype=torch.long)
        elif output_mode == "regression":
            labels = torch.tensor(labels, dtype=torch.float)

        data = (examples, labels)

        if not args.no_cache and not no_cache:
            logger.info(" Saving features into cached file %s" % cached_features_file)
            torch.save(data, cached_features_file)

    return (examples, labels)


class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, args, mode, multi_label, output_mode, no_cache):
        self.examples, self.labels = build_classification_dataset(
            data, tokenizer, args, mode, multi_label, output_mode, no_cache
        )

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return (
            {key: self.examples[key][index] for key in self.examples},
            self.labels[index],
        )


def map_labels_to_numeric(example, multi_label, args):
    if multi_label:
        if isinstance(example["labels"][0], str):
            example["labels"] = [float(1) if i in example["labels"].split(args.labels_sep) else float(0) for i in
                                 args.labels_list]
        else:
            example["labels"] = [float(args.labels_map[label]) for label in example["labels"]]
    else:
        example["labels"] = args.labels_map[example["labels"]]

    return example


def load_hf_dataset(data, tokenizer, args, multi_label):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
    else:
        dataset = HFDataset.from_pandas(data)

    if args.labels_map and not args.regression:
        dataset = dataset.map(lambda x: map_labels_to_numeric(x, multi_label, args))

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x, tokenizer=tokenizer, max_seq_length=args.max_seq_length
        ),
        batched=True,
    )

    if args.model_type in ["bert", "xlnet", "albert", "layoutlm"]:
        dataset.set_format(
            type="pt",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
    else:
        dataset.set_format(type="pt", columns=["input_ids", "attention_mask", "labels"])

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def convert_example_to_feature(
        example_row,
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    if add_prefix_space and not example.text_a.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + example.text_a)
    else:
        tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        if add_prefix_space and not example.text_b.startswith(" "):
            tokens_b = tokenizer.tokenize(" " + example.text_b)
        else:
            tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        if sep_token_extra:
            tokens += [sep_token]
            segment_ids += [sequence_b_segment_id]

        tokens += tokens_b + [sep_token]

        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if pad_to_max_length:
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                                 [0 if mask_padding_with_zero else 1] * padding_length
                         ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label,
    )


def convert_example_to_feature_sliding_window(
        example_row,
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    if stride < 1:
        stride = int(max_seq_length * stride)

    bucket_size = max_seq_length - (3 if sep_token_extra else 2)
    token_sets = []

    if add_prefix_space and not example.text_a.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + example.text_a)
    else:
        tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > bucket_size:
        token_sets = [
            tokens_a[i: i + bucket_size] for i in range(0, len(tokens_a), stride)
        ]
    else:
        token_sets.append(tokens_a)

    if example.text_b:
        raise ValueError(
            "Sequence pair tasks not implemented for sliding window tokenization."
        )

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    input_features = []
    for tokens_a in token_sets:
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                                 [0 if mask_padding_with_zero else 1] * padding_length
                         ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=example.label,
            )
        )

    return input_features


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end=False,
        sep_token_extra=False,
        pad_on_left=False,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 2,
        multi_label=False,
        silent=False,
        use_multiprocessing=True,
        sliding_window=False,
        flatten=False,
        stride=None,
        add_prefix_space=False,
        pad_to_max_length=True,
        args=None,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            multi_label,
            stride,
            pad_token,
            add_prefix_space,
            pad_to_max_length,
        )
        for example in examples
    ]

    if use_multiprocessing:
        if args.multiprocessing_chunksize == -1:
            chunksize = max(len(examples) // (args.process_count * 2), 500)
        else:
            chunksize = args.multiprocessing_chunksize
        if sliding_window:
            with Pool(process_count) as p:
                features = list(
                    tqdm(
                        p.imap(
                            convert_example_to_feature_sliding_window,
                            examples,
                            chunksize=chunksize,
                        ),
                        total=len(examples),
                        disable=silent,
                    )
                )
            if flatten:
                features = [
                    feature for feature_set in features for feature in feature_set
                ]
        else:
            with Pool(process_count) as p:
                features = list(
                    tqdm(
                        p.imap(
                            convert_example_to_feature, examples, chunksize=chunksize
                        ),
                        total=len(examples),
                        disable=silent,
                    )
                )
    else:
        if sliding_window:
            features = [
                convert_example_to_feature_sliding_window(example)
                for example in tqdm(examples, disable=silent)
            ]
            if flatten:
                features = [
                    feature for feature_set in features for feature in feature_set
                ]
        else:
            features = [
                convert_example_to_feature(example)
                for example in tqdm(examples, disable=silent)
            ]

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class LazyClassificationDataset(Dataset):
    def __init__(self, data_file, tokenizer, args):
        self.data_file = data_file
        self.start_row = args.lazy_loading_start_line
        self.num_entries = self._get_n_lines(self.data_file, self.start_row)
        self.tokenizer = tokenizer
        self.args = args
        self.delimiter = args.lazy_delimiter
        if args.lazy_text_a_column is not None and args.lazy_text_b_column is not None:
            self.text_a_column = args.lazy_text_a_column
            self.text_b_column = args.lazy_text_b_column
            self.text_column = None
        else:
            self.text_column = args.lazy_text_column
            self.text_a_column = None
            self.text_b_column = None
        self.labels_column = args.lazy_labels_column

    @staticmethod
    def _get_n_lines(data_file, start_row):
        with open(data_file, encoding="utf-8") as f:
            for line_idx, _ in enumerate(f, 1):
                pass

        return line_idx - start_row

    def __getitem__(self, idx):
        line = (
            linecache.getline(self.data_file, idx + 1 + self.start_row)
            .rstrip("\n")
            .split(self.delimiter)
        )

        if not self.text_a_column and not self.text_b_column:
            text = line[self.text_column]
            label = line[self.labels_column]

            # If labels_map is defined, then labels need to be replaced with ints or floats
            if self.args.labels_map:
                if self.args.multi_label:
                    if isinstance(label[0], str):
                        label = [float(1) if i in label.split(self.args.labels_sep) else float(0) for i in
                                 self.args.labels_map.keys()]
                    else:
                        label = [float(self.args.labels_map[l]) for l in label]
                else:
                    label = self.args.labels_map[label]
            if self.args.regression:
                label = torch.tensor(float(label), dtype=torch.float)
            elif self.args.multi_label:
                label = torch.tensor([float(l) for l in label], dtype=torch.float)
            else:
                label = torch.tensor(int(label), dtype=torch.long)

            return (
                self.tokenizer(
                    text=text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.args.max_seq_length,
                    return_tensors="pt",
                ),
                label,
            )
        else:
            text_a = line[self.text_a_column]
            text_b = line[self.text_b_column]
            label = line[self.labels_column]
            if self.args.regression:
                label = torch.tensor(float(label), dtype=torch.float)
            else:
                label = torch.tensor(int(label), dtype=torch.long)

            return (
                self.tokenizer(
                    text_a,
                    text_pair=text_b,
                    padding="max_length",
                    max_length=self.args.max_seq_length,
                    return_tensors="pt",
                ),
                label,
            )

    def __len__(self):
        return self.num_entries


def flatten_results(results, parent_key="", sep="/"):
    out = []
    if isinstance(results, Mapping):
        for key, value in results.items():
            pkey = parent_key + sep + str(key) if parent_key else str(key)
            out.extend(flatten_results(value, parent_key=pkey).items())
    elif isinstance(results, Iterable):
        for key, value in enumerate(results):
            pkey = parent_key + sep + str(key) if parent_key else str(key)
            out.extend(flatten_results(value, parent_key=pkey).items())
    else:
        out.append((parent_key, results))
    return dict(out)


# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py
# adapted from:
# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/focal.html


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]` for one-vs-others mode (weight of negative class)
                        or :math:`\alpha_i \in \R`
                        vector of weights for each class (analogous to weight argument for CrossEntropyLoss)
        gamma (float): Focusing parameter :math:`\gamma >= 0`. When 0 is equal to CrossEntropyLoss
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’.
         ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
                in the output, uses geometric mean if alpha set to list of weights
         ‘sum’: the output will be summed. Default: ‘none’.
        ignore_index (Optional[int]): specifies indexes that are ignored during loss calculation
         (identical to PyTorch's CrossEntropyLoss 'ignore_index' parameter). Default: -100

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> C = 5  # num_classes
        >>> N = 1 # num_examples
        >>> loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(N, C, requires_grad=True)
        >>> target = torch.empty(N, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(
            self,
            alpha: Optional[Union[float, Iterable]] = 0.5,
            gamma: Real = 2.0,
            reduction: str = "mean",
            ignore_index: int = -100,
            epsilon=1.e-9,
            activation_type='softmax',
    ) -> None:
        super(FocalLoss, self).__init__()
        if (
                alpha is not None
                and not isinstance(alpha, float)
                and not isinstance(alpha, Iterable)
        ):
            raise ValueError(
                f"alpha value should be None, float value or list of real values. Got: {type(alpha)}"
            )
        self.alpha: Optional[Union[float, torch.Tensor]] = (
            alpha
            if alpha is None or isinstance(alpha, float)
            else torch.FloatTensor(alpha)
        )
        if isinstance(alpha, float) and not 0.0 <= alpha <= 1.0:
            warnings.warn("[Focal Loss] alpha value is to high must be between [0, 1]")

        self.gamma: Real = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index
        self.epsilon = epsilon
        self.activation_type: str = activation_type

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the criterion.
        @param input: model's output, shape [N, C] meaning N: batch_size and C: num_classes
        @param target: ground truth, shape [N], N: batch_size
        @return: shape of [N]
        """
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if input.device != target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )

        # filter labels
        target = target.type(torch.long)

        if self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            focal_loss = -self.alpha * multi_hot_key * \
                         torch.pow((1 - logits), self.gamma) * \
                         (logits + self.epsilon).log()
            focal_loss += -(1 - self.alpha) * zero_hot_key * \
                          torch.pow(logits, self.gamma) * \
                          (1 - logits + self.epsilon).log()
            weights = torch.ones_like(
                focal_loss, dtype=focal_loss.dtype, device=focal_loss.device
            )
        else:
            input_mask = target != self.ignore_index
            target = target[input_mask]
            input = input[input_mask]
            # compute softmax over the classes axis
            pt = F.softmax(input, dim=1)
            logpt = F.log_softmax(input, dim=1)

            # compute focal loss
            pt = pt.gather(1, target.unsqueeze(-1)).squeeze()
            logpt = logpt.gather(1, target.unsqueeze(-1)).squeeze()
            focal_loss = -1 * (1 - pt) ** self.gamma * logpt

            weights = torch.ones_like(
                focal_loss, dtype=focal_loss.dtype, device=focal_loss.device
            )
            if self.alpha is not None:
                if isinstance(self.alpha, float):
                    alpha = torch.tensor(self.alpha, device=input.device)
                    weights = torch.where(target > 0, 1 - alpha, alpha)
                elif torch.is_tensor(self.alpha):
                    alpha = self.alpha.to(input.device)
                    weights = alpha.gather(0, target)

        tmp_loss = focal_loss * weights
        if self.reduction == "none":
            loss = tmp_loss
        elif self.reduction == "mean":
            loss = (
                tmp_loss.sum() / weights.sum()
                if torch.is_tensor(self.alpha)
                else torch.mean(tmp_loss)
            )
        elif self.reduction == "sum":
            loss = tmp_loss.sum()
        else:
            raise NotImplementedError(
                "Invalid reduction mode: {}".format(self.reduction)
            )
        return loss


def init_loss(weight, device, args, multi_label=False):
    """ 
    weighted loss or focal loss, common loss is defined in the model definition 
    """
    if weight and args.loss_type:
        warnings.warn(
            f"weight and args.loss_type parameters are set at the same time"
            f"will use weighted cross entropy loss. To use {args.loss_type} set weight to None"
        )
    if weight:
        if multi_label:
            loss_fct = nn.BCEWithLogitsLoss(weight=torch.Tensor(weight).to(device))
        else:
            loss_fct = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
    elif args.loss_type:
        if args.loss_type == "focal":
            if multi_label:
                loss_fct = FocalLoss(activation_type='sigmoid', **args.loss_args)
            else:
                loss_fct = FocalLoss(activation_type='softmax', **args.loss_args)
        else:
            raise NotImplementedError(f"unknown {args.loss_type} loss function")
    else:
        loss_fct = None

    return loss_fct
