import numpy as np
import math
import logging


class EarlyStopping:
    def __init__(self, name, min_delta=0, patience=0, mode='min', baseline=None):
        self.name = name
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.mode = mode
        self.baseline = baseline
        self.best_model_params = None
        self.best_metrics = None

        if mode not in ['min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, fallback to min mode.', mode)
            mode = 'min'

        if mode == 'min':
            self.compare_op = np.less_equal
        else:
            self.compare_op = np.greater_equal

        if self.mode == 'min':
            self.min_delta *= -1

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.mode == 'min' else -np.Inf

    def should_stop(self, value, model_params=None, metrics=None):
        if self.compare_op(value - self.min_delta, self.best):
            self.best = value
            self.best_model_params = model_params
            self.best_metrics = metrics
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

    def get_best_model_params(self):
        return self.best_model_params

    def get_best_metrics(self):
        return self.best_metrics


def get_maybe_perform_early_stopping_fn(cfg):
    early_stoppers = {}
    if cfg.training.get('early_stopping') is not None:
        for early_stopper_name in cfg.training.early_stopping:
            early_stopper_cfg = cfg.training.early_stopping[early_stopper_name]
            early_stopper = EarlyStopping(name=early_stopper_name,
                                          min_delta=early_stopper_cfg.min_delta,
                                          patience=math.ceil(early_stopper_cfg.patience / cfg.evaluation.rounds_per_eval),
                                          mode=early_stopper_cfg.mode,
                                          baseline=early_stopper_cfg.get('baseline', None))
            early_stoppers[early_stopper_name] = early_stopper

    def maybe_perform_early_stopping(metrics_dict, round_num, model_weights):
        for early_stopper_name, early_stopper in early_stoppers.items():
            early_stopper_cfg = cfg.training.early_stopping[early_stopper_name]
            if round_num >= early_stopper_cfg.initial_delay:
                if early_stopper_cfg.key in metrics_dict.keys():
                    if early_stopper.should_stop(metrics_dict[early_stopper_cfg.key], model_weights, metrics_dict):
                        logging.info(
                            f'Early stopping in epoch {round_num} due to failure to improve {early_stopper_name}')
                        return early_stopper
        return None

    return maybe_perform_early_stopping
