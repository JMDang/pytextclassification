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
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    cosine_schedule_num_cycles: float = 0.5
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)
    dataloader_num_workers: int = 0
    do_lower_case: bool = False
    dynamic_quantize: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "f1"
    early_stopping_metric_minimize: bool = False
    early_stopping_patience: int = 4
    encoding: str = None
    batch_size: int = 128
    evaluate_during_training: bool = True
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-5
    local_rank: int = -1
    logging_steps: int = 50
    loss_type: str = None
    loss_args: dict = field(default_factory=dict)
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    model_name: str = None
    model_type: str = None
    num_labels: int = 2
    use_cuda: bool = True
    multiprocessing_chunksize: int = -1
    n_gpu: int = 1
    no_cache: bool = True
    no_save: bool = False
    not_saved_args: list = field(default_factory=list)
    num_train_epochs: int = 1
    optimizer: str = "AdamW"
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0
    process_count: int = field(default_factory=get_default_process_count)
    quantized_model: bool = False
    reprocess_input_data: bool = True
    save_best_model: bool = True
    save_model_every_epoch: bool = True
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 20
    scheduler: str = "linear_schedule_with_warmup"
    silent: bool = False
    skip_special_tokens: bool = True
    tensorboard_dir: str = "log/"
    thread_count: int = None
    tokenizer_name: str = None
    tokenizer_type: str = None
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_hf_datasets: bool = False
    use_multiprocessing: bool = False
    use_multiprocessing_for_evaluation: bool = False
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0
    onnx: bool = False

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
        if "settings" in args_for_saving["wandb_kwargs"]:
            del args_for_saving["wandb_kwargs"]["settings"]
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            if args_dict["tokenizer_type"] is not None and not isinstance(
                    args_dict["tokenizer_type"], str
            ):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__
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
    lazy_delimiter: str = "\t"
    lazy_labels_column: int = 1
    lazy_loading: bool = False
    lazy_loading_start_line: int = 1
    lazy_text_a_column: bool = None
    lazy_text_b_column: bool = None
    lazy_text_column: int = 0
    sliding_window: bool = False
    special_tokens_list: list = field(default_factory=list)
    stride: float = 0.8
    tie_value: int = 1
    regression: bool = False
    threshold: float = 0.5
    labels_sep: str = ","
