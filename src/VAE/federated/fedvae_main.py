import importlib
import logging
import math
import os
import time
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf

from src.VAE import losses as vae_losses
from src.VAE.federated import fedvae_tff
from src.VAE.vae_train_fns import create_vae_train_fn
from src.data.abstract_dataset import AbstractDataset
from src.differential_privacy import GaussianNoiseNormalizedQuery, L2ClipNormalizedQuery, RDPAccountant, \
    calculate_dp_from_rdp, dp_utils
from src.metrics import get_metrics_list
from src.utils import optimizer_utils, tf_utils
from src.utils.checkpointing import get_write_model_checkpoint_fn
from src.utils.early_stopping import get_maybe_perform_early_stopping_fn
from src.utils.plot_utils import plot_2d_latent_space
from src.utils.progress import Progress


def run_single_trial(dataset: AbstractDataset, eval_hook_fn, cfg):
    model_module = importlib.import_module(f'.{cfg.model.module}', package='src.VAE.models')
    impl_module = importlib.import_module(f'.{cfg.model.type}', package='src.VAE.federated')

    # Setup rdp_accountant
    rdp_accountant, max_local_steps = setup_rdp_accountant(cfg, dataset)

    train_data = dataset.train_ds

    # Initialize client states.
    initial_client_states = impl_module.get_initial_client_states(dataset.client_ids, max_local_steps, model_module, cfg)

    # Get client optimizer functions
    client_encoder_optimizer_fn, client_decoder_optimizer_fn = get_client_optimizer_fns(cfg)

    vae = create_vae_fns(
        vae_model_fn=partial(model_module.define_vae, cfg.model.latent_dim, cfg.dataset.data_dim, cfg.dataset.data_ch, cfg.dataset.n_classes, cfg.model.output_activation_fn),
        vae_loss_fn=vae_losses.get_loss_fn(cfg.training.vae_loss_fn),
        client_encoder_optimizer_fn=client_encoder_optimizer_fn,
        client_decoder_optimizer_fn=client_decoder_optimizer_fn,
        aggregation_optimizer_fn=lambda: optimizer_utils.get_optimizer(cfg.training.aggregation_step_optimizer),
        client_state_fn=lambda: initial_client_states[dataset.client_ids[0]],
        dummy_model_input=next(iter(train_data.create_tf_dataset_for_client(dataset.client_ids[0]))),
        dp_config=cfg.differential_privacy,
        n_clients_per_round=math.floor(cfg.dataset.n_clients * cfg.training.clients_per_round),
        impl_class=impl_module
    )
    # Prepare iterative process
    iterative_process = fedvae_tff.build_federated_vae_process(vae, impl_module)
    initial_server_state = iterative_process.initialize()

    # Prepare callback fns
    get_initial_state, write_model_checkpoint = get_write_model_checkpoint_fn(
        disable_checkpointing=(cfg.model.type == 'sync_d'))

    # get initial state, possibly from disk
    server_state, client_states, start_round = get_initial_state(initial_server_state, initial_client_states)

    # Save model weight names for histograms
    eval_model = vae.model_fn()

    # setup metrics dataframe
    if os.path.exists('metrics.csv'):
        # resuming a run
        metrics = pd.read_csv('metrics.csv', index_col='global_round')
    else:
        metrics = pd.DataFrame(columns=get_metrics_list(cfg))

    maybe_perform_early_stopping = get_maybe_perform_early_stopping_fn(cfg)
    progress = Progress(cfg.training.total_rounds, cfg.evaluation.rounds_per_eval)

    # Run training loop
    logging.info('=================== TRAINING PHASE ==================')

    def run_eval():
        eval_start_time = time.time()
        impl_module.set_eval_model_weights(eval_model, server_state.model_weights)
        eval_metrics = eval_hook_fn(eval_model,
                                    round_num,
                                    partial(_vae_specific_eval_fn,
                                            cfg.model.latent_dim, cfg.dataset.n_classes, cfg.dataset.data_dim,
                                            f'images/round_{round_num}-latent_space'))
        progress.add_eval_duration(time.time() - eval_start_time)
        return eval_metrics

    active_clients_list = dataset.client_ids[:]
    for round_num in range(start_round, cfg.training.total_rounds + 1):
        # =================================
        # ========== ROUND SETUP ==========
        # =================================
        round_start_time = time.time()

        # Sample client data and states
        sampled_clients = tf_utils.sample_clients(active_clients_list, cfg.training.clients_per_round)
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]
        sampled_client_states = [client_states[client] for client in sampled_clients]
        metrics.loc[round_num, 'n_sampled_clients'] = len(sampled_clients)

        # ================================
        # ========== TRAIN STEP ==========
        # ================================
        server_state, train_metrics, updated_client_states, client_weights_delta, weights_delta_aggregate = iterative_process.next(
            server_state, sampled_train_data, sampled_client_states)

        metrics.loc[round_num, 'train_loss'] = train_metrics['loss']

        # check for nan or inf
        if any([(math.isnan(metric) or math.isinf(metric)) for metric in train_metrics.values()]):
            logging.info(f'Stopping in Epoch {round_num} due to NaN')
            break

        # Save updated client states back into the global `client_states` structure.
        for client_state in updated_client_states:
            client_id = dataset.client_ids[client_state.client_index]
            client_states[client_id] = client_state
            logging.debug(f'{client_id}: iterations: {client_state.iters_count}')
            # check for exceeding of client privacy budget
            if cfg.differential_privacy.type == 'local' and client_state.iters_count >= client_state.max_iters:
                active_clients_list.remove(client_id)
                logging.debug(f'removed {client_id} from active list due to exceeding of max local optimisation '
                              f'steps ({len(active_clients_list)} clients left)')

        # ========================================
        # ========== LOGGING AND CHECKS ==========
        # ========================================
        if cfg.evaluation.plot_l2_norms:
            def reduce_median(v):
                v = tf.reshape(v, [-1])
                mid = v.get_shape()[0] // 2 + 1
                return tf.nn.top_k(v, mid).values[-1]

            metrics.loc[round_num, 'L2-Norms/mean_client_delta'] = tf.reduce_mean(
                [tf.linalg.global_norm(tf.nest.flatten(delta)) for delta in client_weights_delta]).numpy()
            metrics.loc[round_num, 'L2-Norms/median_client_delta'] = reduce_median(
                [tf.linalg.global_norm(tf.nest.flatten(delta)) for delta in client_weights_delta]).numpy()
            metrics.loc[round_num, 'L2-Norms/client_delta_aggregate'] = tf.linalg.global_norm(
                tf.nest.flatten(weights_delta_aggregate)).numpy()

        if len(active_clients_list) == 0:
            logging.info(f'Early stopping due to empty client list (all local privacy budgets spent)')
            # Perform final evaluation
            eval_metrics = run_eval()
            for key, value in eval_metrics.items():
                metrics.loc[round_num, key] = value
            break

        # ===== Log current privacy spending and l2_norm_clip/stddev setup =====
        if cfg.differential_privacy.type == 'central':
            if cfg.differential_privacy.adaptive:
                sum_state = server_state.aggregation_state[0].numerator_state.sum_state
            else:
                sum_state = server_state.aggregation_state[0].numerator_state
            epsilon, opt_order = rdp_accountant.get_privacy_spending_for_n_steps(round_num)
            metrics.loc[round_num, 'RDP/epsilon'] = epsilon
            metrics.loc[round_num, 'RDP/opt_order'] = opt_order

            if sum_state.stddev >= 10.0:
                # If adaptive DP diverges, stop early
                logging.info(f'Early stopping due to too high noise introduction (stddev = {sum_state.stddev})')
                # Perform final evaluation
                eval_metrics = run_eval()
                for key, value in eval_metrics.items():
                    metrics.loc[round_num, key] = value
                break

        logging.debug(
            f'Round {round_num} total iterations on sampled clients: {server_state.counters["total_iters_count"]}')

        # Write model checkpoint
        if cfg.training.rounds_per_checkpoint > 0 and (round_num % cfg.training.rounds_per_checkpoint == 0 or
                                                       round_num == cfg.training.total_rounds):
            write_model_checkpoint(round_num, server_state, client_states)

        # Save round duration
        round_train_duration = time.time() - round_start_time
        progress.add_train_duration(round_train_duration)

        # ================================
        # ========== EVALUATION ==========
        # ================================
        if round_num % cfg.evaluation.rounds_per_eval == 0 or round_num == cfg.training.total_rounds:
            eval_metrics = run_eval()
            for key, value in eval_metrics.items():
                metrics.loc[round_num, key] = value

            # check for early stopping criteria
            triggered_early_stopper = maybe_perform_early_stopping(eval_metrics, round_num, server_state.model_weights)
            if triggered_early_stopper is not None:
                if triggered_early_stopper.get_best_metrics() is not None:
                    # Override latest eval metrics with stored best ones
                    for key, value in triggered_early_stopper.get_best_metrics().items():
                        metrics.loc[round_num, key] = value
                    server_state.model_weights = triggered_early_stopper.get_best_model_params()
                break

        # Log round info
        eta_hours, eta_minutes = progress.eta(round_num)
        logging.info(
            f'Round {round_num}/{cfg.training.total_rounds} took {str(timedelta(seconds=round_train_duration))[2:-7]} -'
            f' {eta_hours} hrs {eta_minutes} mins remaining')

        # check for early stopping criteria
        triggered_early_stopper = maybe_perform_early_stopping(train_metrics, round_num, server_state.model_weights)
        if triggered_early_stopper is not None:
            if triggered_early_stopper.get_best_metrics() is not None:
                # Override latest eval metrics with stored best ones
                for key, value in triggered_early_stopper.get_best_metrics().items():
                    metrics.loc[round_num, key] = value
                server_state.model_weights = triggered_early_stopper.get_best_model_params()
            break

        # Save metrics DF
        metrics.to_csv('metrics.csv', index_label='global_round')

    # Save metrics DF again (in case we hit a "break")
    metrics.to_csv('metrics.csv', index_label='global_round')

    final_model = vae.model_fn()
    impl_module.set_eval_model_weights(final_model, server_state.model_weights)

    return metrics, final_model.decoder


def setup_rdp_accountant(cfg, dataset):
    if cfg.differential_privacy.type == 'local':
        start_time = time.time()
        rdp_accountants = {
            client_id: RDPAccountant(
                q=min(1.0, cfg.training.batch_size / dataset.get_dataset_size_for_client(client_id)),
                z=cfg.differential_privacy.noise_multiplier,
                N=dataset.get_dataset_size_for_client(client_id),
                dp_type=cfg.differential_privacy.type,
                max_eps=cfg.differential_privacy.epsilon,
                target_delta=cfg.differential_privacy.delta)
            for client_id in dataset.client_ids}
        mid_time = time.time()
        logging.debug(f'Setting up local RDP Accountants took {mid_time - start_time:.3f}s')
        local_steps_cache = {}

        def cached_max_local_steps(rdp_accountant: RDPAccountant):
            if rdp_accountant.q in local_steps_cache.keys():
                return local_steps_cache[rdp_accountant.q]
            max_n_steps, _, _ = rdp_accountant.get_maximum_n_steps(base_n_steps=0,
                                                                   n_steps_increment=10)
            local_steps_cache[rdp_accountant.q] = max_n_steps
            return max_n_steps

        max_local_steps = list(map(cached_max_local_steps, rdp_accountants.values()))
        logging.info(f'Maximum local steps: {np.mean(max_local_steps)} +/- {np.std(max_local_steps)}')
        logging.debug(f'Calculating max local steps took {time.time() - mid_time:.3f}s')
        if np.mean(max_local_steps) < 1.0:
            logging.info(f'Stopping run due to no local privacy budget.')
            exit(0)
        return rdp_accountants, max_local_steps

    if cfg.differential_privacy.type == 'central':
        rdp_accountant = RDPAccountant(q=cfg.training.clients_per_round,
                                       z=cfg.differential_privacy.noise_multiplier,
                                       N=cfg.dataset.n_clients,
                                       dp_type=cfg.differential_privacy.type,
                                       max_eps=cfg.differential_privacy.epsilon,
                                       target_delta=cfg.differential_privacy.delta)
        noise_std = rdp_accountant.compute_actual_noise_std(S=cfg.differential_privacy.l2_norm_clip)
        logging.info(f'DP noise STD: {noise_std:.3f}')
        max_rounds_for_dp, eps, opt_order = rdp_accountant.get_maximum_n_steps(max_n_steps=cfg.training.total_rounds)

        if cfg.training.total_rounds > max_rounds_for_dp:
            logging.info(f'Reducing total_rounds to {max_rounds_for_dp} to keep privacy budget')
            cfg.training.total_rounds = max_rounds_for_dp
        logging.info(f'Training follows RDP with epsilon={eps:.2f} (alpha={opt_order}) - equivalent to classical DP ' +
                     f'with epsilon={calculate_dp_from_rdp(opt_order, eps, cfg.differential_privacy.delta)[0]:.2f} ' +
                     f'and delta={cfg.differential_privacy.delta})')
        return rdp_accountant, [-1] * cfg.dataset.n_clients
    return None, [-1] * cfg.dataset.n_clients


def get_client_optimizer_fns(cfg):
    if cfg.differential_privacy.type == 'local':
        if cfg.model.type == 'sync_d':
            # Only use DP optimizer for decoder, which is synchronised
            client_encoder_optimizer_fn = lambda: optimizer_utils.get_optimizer(cfg.training.client_optimizer)
            if cfg.training.get('client_dec_optimizer') is not None:
                client_decoder_optimizer_fn = lambda: optimizer_utils.get_dp_optimizer(cfg.training.client_dec_optimizer,
                                                                                       cfg.differential_privacy,
                                                                                       cfg.training.batch_size)
            else:
                client_decoder_optimizer_fn = lambda: optimizer_utils.get_dp_optimizer(cfg.training.client_optimizer,
                                                                                       cfg.differential_privacy,
                                                                                       cfg.training.batch_size)
        else:
            client_encoder_optimizer_fn = lambda: optimizer_utils.get_dp_optimizer(cfg.training.client_optimizer,
                                                                                   cfg.differential_privacy,
                                                                                   cfg.training.batch_size)
            if cfg.training.get('client_dec_optimizer') is not None:
                client_decoder_optimizer_fn = lambda: optimizer_utils.get_dp_optimizer(cfg.training.client_dec_optimizer,
                                                                                       cfg.differential_privacy,
                                                                                       cfg.training.batch_size)
            else:
                client_decoder_optimizer_fn = client_encoder_optimizer_fn
    else:
        client_encoder_optimizer_fn = lambda: optimizer_utils.get_optimizer(cfg.training.client_optimizer)
        if cfg.training.get('client_dec_optimizer') is not None:
            client_decoder_optimizer_fn = lambda: optimizer_utils.get_optimizer(cfg.training.client_dec_optimizer)
        else:
            client_decoder_optimizer_fn = client_encoder_optimizer_fn

    return client_decoder_optimizer_fn, client_encoder_optimizer_fn


def create_vae_fns(vae_model_fn, vae_loss_fn,
                   client_encoder_optimizer_fn, client_decoder_optimizer_fn, aggregation_optimizer_fn,
                   client_state_fn, dummy_model_input, dp_config, n_clients_per_round, impl_class):
    train_vae_fn = create_vae_train_fn(vae_loss_fn,
                                       encoder_optimizer=client_encoder_optimizer_fn(),
                                       decoder_optimizer=client_decoder_optimizer_fn(),
                                       local_dp=dp_config.type == 'local')

    dp_average_query = None
    if dp_config.type == 'central':
        dp_average_query = dp_utils.get_dp_query(noise_multiplier=dp_config.noise_multiplier,
                                                 n_clients_per_round=n_clients_per_round,
                                                 l2_norm_clip=dp_config.l2_norm_clip,
                                                 adaptive=dp_config.adaptive)
    elif dp_config.type == 'noise_only':
        dp_average_query = GaussianNoiseNormalizedQuery(stddev=dp_config.effective_noise,
                                                        denominator=n_clients_per_round)
    elif dp_config.type == 'clip_only':
        dp_average_query = L2ClipNormalizedQuery(l2_norm_clip=dp_config.l2_norm_clip,
                                                 denominator=n_clients_per_round)

    return fedvae_tff.VaeFnsAndTypes(
        model_fn=vae_model_fn,
        aggregation_optimizer_fn=aggregation_optimizer_fn,
        client_state_fn=client_state_fn,
        train_vae_fn=train_vae_fn,
        dummy_model_input=dummy_model_input,
        dp_average_query=dp_average_query,
        sync_decoder_only=impl_class.sync_decoder_only
    )


def _vae_specific_eval_fn(latent_dim, n_classes, data_dim, base_file_name, model):
    if latent_dim == 2:
        os.makedirs(base_file_name, exist_ok=True)
        for i in range(n_classes):
            plot_2d_latent_space(model, i, data_dim, file_name=f'{base_file_name}/label_{i}.png')
