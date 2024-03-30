# -*- coding: utf-8 -*-
"""
author:dangjinming(jmdang777@qq.com)
"""
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from multiprocessing import cpu_count
import warnings

from torch.utils.data import Dataset


def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count

def get_special_tokens():
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

@dataclass
class ModelArgs:
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model/"
    early_stopping_delta: float = 0
    early_stopping_metric: str = "f1"
    early_stopping_metric_minimize: bool = False
    early_stopping_patience: int = 4
    batch_size: int = 128
    evaluate_during_training: bool = True
    evaluate_during_training_steps: int = 10
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-3
    max_grad_norm: float = 1.0
    max_seq_length: int = 100
    num_labels: int = 2
    use_cuda: bool = True
    no_save: bool = False
    num_train_epochs: int = 3
    not_saved_args: list = field(default_factory=list)
    optimizer: str = "AdamW"
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    save_best_model: bool = True
    save_model_every_epoch: bool = True
    save_optimizer: bool = True
    save_steps: int = 20
    silent: bool = False
    tensorboard_dir: str = "log/"
    logging_steps: int = 50
    use_early_stopping: bool = False
    weight_decay: float = 0.0

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {
            key: value
            for key, value in asdict(self).items()
            if key not in self.not_saved_args
        }
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class ClassificationArgs(ModelArgs):
    """
    Model args for a ClassificationModel
    """ 
    model_class: str = "ClassificationModel"
    multi_label:bool = False
    labels_list: list = field(default_factory=list)
    labels_map: dict = field(default_factory=dict)
    threshold: float = 0.5
    regression: bool = False
    labels_sep: str = ","
    vocab_level: str = "word"
    embed_size: int = 300
    max_vocab_size: int = 200000
    SEED: int = 12345
    pooling_type: str = "mean"
    
    #RNN
    hidden_size: int = 256
    num_layers: int = 2
    lstm_dropout_rate: float = 0.5
    lstm_hidden_size: int = 256
    #CNN
    num_filters:int = 256
    cnn_dropout_rate: float = 0.5
    filter_sizes: tuple = field(default_factory=tuple)
    #fasttext
    enable_ngram: bool = True
    n_gram_buckets_size: int = 700000
    fast_text_hidden_size: int = 256
    fast_text_dropout_rate: float = 0.5
    #machine learing
    model_name_or_model: str = "lr"
    feature_name_or_feature: str = "tfidf"
    stopwords_path: str = "data/stopwords.txt"