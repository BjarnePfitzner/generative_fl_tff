import importlib
import logging
import math
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from src.VAE.losses import get_loss_fn
from src.VAE.vae_train_fns import create_vae_train_fn
from src.data.abstract_dataset import AbstractDataset
from src.differential_privacy import RDPAccountant, calculate_dp_from_rdp
from src.metrics import get_metrics_list
from src.utils import optimizer_utils
from src.utils.early_stopping import get_maybe_perform_early_stopping_fn
from src.utils.progress import Progress


def run_single_trial(dataset: AbstractDataset, eval_hook_fn, cfg):
    model_module = importlib.import_module(f'.{cfg.model.module}', package='src.VAE.models')
    model = model_module.define_vae(cfg.model.latent_dim, cfg.dataset.data_dim, cfg.dataset.data_ch, cfg.dataset.n_classes)

    # Setup rdp_accountant
    rdp_accountant, max_steps_for_dp = setup_rdp_accountant(cfg, dataset)
    if cfg.differential_privacy.type != 'disabled':
        # Setup DP optimizers
        enc_optimizer = optimizer_utils.get_dp_optimizer(cfg.training.client_optimizer, cfg.differential_privacy,
                                                         cfg.training.batch_size)
        dec_optimizer = optimizer_utils.get_dp_optimizer(cfg.training.client_optimizer, cfg.differential_privacy,
                                                         cfg.training.batch_size)

        # Print DP information
        l2_norm_clip = cfg.differential_privacy.l2_norm_clip
        noise_std = l2_norm_clip * cfg.differential_privacy.noise_multiplier
        eff_noise_std = rdp_accountant.compute_actual_noise_std(l2_norm_clip)
        distortion = (l2_norm_clip ** 2) * ((noise_std) ** 2) / cfg.training.batch_size
        logging.info(f'Running experiment with L2 norm clip {l2_norm_clip} and noise std.dev. {noise_std} (effective ' +
                     f'noise std.dev. {eff_noise_std}), resulting in a distortion of {distortion}.')
    else:
        enc_optimizer = optimizer_utils.get_optimizer(cfg.training.client_optimizer)
        dec_optimizer = optimizer_utils.get_optimizer(cfg.training.client_optimizer)

    vae_loss_fn = get_loss_fn(cfg.training.vae_loss_fn)
    train_vae_fn = tf.function(create_vae_train_fn(vae_loss_fn,
                                                   encoder_optimizer=enc_optimizer, decoder_optimizer=dec_optimizer,
                                                   local_dp=cfg.differential_privacy.type != 'disabled'))

    # setup metrics dataframe
    if os.path.exists(f'{cfg.run_dir}/metrics.csv'):
        # resuming a run
        metrics = pd.read_csv(f'{cfg.run_dir}/metrics.csv', index_col='global_round')
    else:
        metrics = pd.DataFrame(columns=get_metrics_list(cfg))

    maybe_perform_early_stopping = get_maybe_perform_early_stopping_fn(cfg)
    progress = Progress(cfg.training.total_rounds, cfg.evaluation.rounds_per_eval)

    def run_eval():
        eval_start_time = time.time()
        eval_metrics = eval_hook_fn(model, round_num)
        progress.add_eval_duration(time.time() - eval_start_time)
        return eval_metrics

    logging.info('Starting training loop.')
    n_steps = 0
    for round_num in range(1, cfg.training.total_rounds + 1):
        round_start_time = time.time()
        loss_dict = None

        # ========== TRAIN STEP ==========
        for images, labels in dataset.train_ds:
            batch_size, batch_losses, _ = train_vae_fn(model, images, labels, global_round=tf.constant(round_num))
            if loss_dict is None:
                loss_dict = {key: [value] for key, value in batch_losses.items()}
            else:
                loss_dict = {key: loss_dict[key] + [value] for key, value in batch_losses.items()}
            n_steps += 1
            if cfg.differential_privacy.type != 'disabled' and n_steps >= max_steps_for_dp:
                break

        train_metrics = {key: np.mean(the_losses) for key, the_losses in loss_dict.items() if len(the_losses) > 0}

        # check for nan
        if any([(math.isnan(metric) or math.isinf(metric)) for metric in train_metrics.values()]):
            logging.info(f'Early Stopping in Epoch {round_num} due to NaN')
            break

        metrics.loc[round_num, 'train_loss'] = train_metrics['loss']

        round_train_duration = time.time() - round_start_time
        progress.add_train_duration(round_train_duration)

        if cfg.differential_privacy.type != 'disabled':
            epsilon, opt_order = rdp_accountant.get_privacy_spending_for_n_steps(round_num / rdp_accountant.q)
            metrics.loc[round_num, 'RDP/epsilon'] = epsilon
            metrics.loc[round_num, 'RDP/opt_order'] = opt_order

        # Evaluation
        if round_num % cfg.evaluation.rounds_per_eval == 0 or round_num == cfg.training.total_rounds:
            eval_metrics = run_eval()
            for key, value in eval_metrics.items():
                metrics.loc[round_num, key] = value

            # check for early stopping criteria
            triggered_early_stopper = maybe_perform_early_stopping(eval_metrics, round_num, model.get_weights())
            if triggered_early_stopper is not None:
                if triggered_early_stopper.get_best_metrics() is not None:
                    # Override latest eval metrics with stored best ones
                    for key, value in triggered_early_stopper.get_best_metrics().items():
                        metrics.loc[round_num, key] = value
                    model.set_weights(triggered_early_stopper.get_best_model_params())
                break

        # Log round info
        eta_hours, eta_minutes = progress.eta(round_num)
        logging.info(
            f'Round {round_num}/{cfg.training.total_rounds} took {str(timedelta(seconds=round_train_duration))} - '
            f'{eta_hours} hrs {eta_minutes} mins remaining')

        # check for early stopping criteria
        triggered_early_stopper = maybe_perform_early_stopping(train_metrics, round_num, model.get_weights())
        if triggered_early_stopper is not None:
            if triggered_early_stopper.get_best_metrics() is not None:
                # Override latest eval metrics with stored best ones
                for key, value in triggered_early_stopper.get_best_metrics().items():
                    metrics.loc[round_num, key] = value
                model.set_weights(triggered_early_stopper.get_best_model_params())
            break

        # Save metrics DF
        metrics.to_csv(f'{cfg.run_dir}/metrics.csv', index_label='global_round')

    # Save metrics DF again (in case we hit a "break")
    metrics.to_csv(f'{cfg.run_dir}/metrics.csv', index_label='global_round')

    return metrics, model.decoder


def setup_rdp_accountant(cfg, dataset):
    if cfg.differential_privacy.type == 'disabled':
        return None, -1
    rdp_accountant = RDPAccountant(q=cfg.training.batch_size / dataset.dataset_size['train'],
                                   z=cfg.differential_privacy.noise_multiplier,
                                   N=dataset.dataset_size['train'],
                                   dp_type='local',
                                   max_eps=cfg.differential_privacy.epsilon,
                                   target_delta=cfg.differential_privacy.delta)
    noise_std = rdp_accountant.compute_actual_noise_std(S=cfg.differential_privacy.l2_norm_clip)
    logging.info(f'DP noise STD: {noise_std:.3f}')

    # Compute maximum allowed steps
    max_steps_for_dp, eps, opt_order = rdp_accountant.get_maximum_n_steps(
        max_n_steps=cfg.training.total_rounds / rdp_accountant.q)
    max_rounds_for_dp = math.ceil(max_steps_for_dp * rdp_accountant.q)
    if cfg.training.total_rounds > max_rounds_for_dp:
        logging.info(f'Reducing total_rounds to {max_rounds_for_dp} to keep privacy budget')
        cfg.training.total_rounds = max_rounds_for_dp
    logging.info(f'Training for {max_steps_for_dp} optimisation steps follows ({opt_order}, {eps:.2f})-RDP (equivalent to '
                 f'({calculate_dp_from_rdp(opt_order, eps, cfg.differential_privacy.delta)[0]:.2f}, '
                 f'{cfg.differential_privacy.delta})-DP')

    return rdp_accountant, max_steps_for_dp
