import importlib
import logging
import math
import os

import hydra
import tensorflow as tf
import tensorflow_federated as tff
from omegaconf import DictConfig, OmegaConf

from src.data import get_dataset_instance
from src.evaluation import get_eval_hook_fn


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    tff.backends.native.set_local_execution_context(clients_per_thread=1,
                                                    client_tf_devices=tf.config.list_logical_devices('GPU'))

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.info('==================== SETUP PHASE ====================')

    # ========== Setup run directory =========
    if os.path.exists('hydra_config.yaml'):
        if cfg.resume_runs:
            logging.info('Resumung run')
            logging.info('loading config file...')
            with open('hydra_config.yaml', 'r') as f:
                cfg = OmegaConf.load(f)
        else:
            logging.info('Run directory not empty. Exiting...')
            exit(0)
    os.makedirs('images', exist_ok=True)

    # ========== Load dataset ==========
    dataset = get_dataset_instance(cfg)

    # ========== load main function ==========
    if cfg.model.type == 'centralized':
        main_module = importlib.import_module(f'.{cfg.model.name}_main', package=f'src.{cfg.model.name.upper()}.centralized')
    else:
        main_module = importlib.import_module(f'.fed{cfg.model.name}_main', package=f'src.{cfg.model.name.upper()}.federated')

        if cfg.adjust_total_rounds:
            previous_total_rounds = cfg.training.total_rounds
            cfg.training.total_rounds = round(cfg.training.total_rounds / math.sqrt(cfg.training.clients_per_round * cfg.training.local_epochs))
            if cfg.training.total_rounds != previous_total_rounds:
                logging.info(f'Adjusting total rounds to {cfg.training.total_rounds} from {previous_total_rounds}' +
                             f' due to {cfg.training.clients_per_round} clients/round and {cfg.training.local_epochs}' +
                             ' local epochs')
            if cfg.training.get('early_stopping') is not None:
                for early_stopper_name in cfg.training.early_stopping:
                    factor = min(2.0, max(1.0, 1 / (cfg.training.clients_per_round * cfg.training.local_epochs)))
                    if factor > 1.0:
                        new_patience = round(cfg.training.early_stopping[early_stopper_name].patience * factor)
                        cfg.training.early_stopping[early_stopper_name].patience = new_patience
                        logging.info(f'Adjusting {early_stopper_name} early stopper patience to {new_patience}')

    if cfg.training.get('early_stopping.accuracy') is not None:
        acc_baseline = round(dataset.get_default_accuracy() / 2, 2)
        logging.info(f'Setting accuracy early stopping baseline to {acc_baseline}')
        cfg.training.early_stopping.accuracy.baseline = acc_baseline

    if cfg.evaluation.run_cm_eval:
        logging.info('Setting cfg.evaluation.real_classifier to True to allow for confusion matrix evaluation.')
        cfg.evaluation.real_classifier = True

    dataset.preprocess_datasets(cfg.training.batch_size, cfg.evaluation.batch_size, cfg.training.local_epochs,
                                local_dp=cfg.differential_privacy.type == 'local')    # Required for local DP optimizer microbatch to function properly
    if cfg.differential_privacy.type == 'local':
        cfg.dataset.n_clients = len(dataset.client_ids)

    # ========== Save config file ==========
    yaml = OmegaConf.to_yaml(cfg)
    logging.info(yaml)
    with open('hydra_config.yaml', 'w') as f:
        f.write(yaml)

    # ========================================
    # ========== Setup eval hook fn ==========
    # ========================================
    seed_noise = tf.random.normal([cfg.dataset.n_generated_per_class * cfg.dataset.n_classes, cfg.model.latent_dim])
    seed_labels = tf.repeat(tf.range(0, limit=cfg.dataset.n_classes), cfg.dataset.n_generated_per_class)
    seed = [seed_noise, seed_labels]
    eval_hook_fn = get_eval_hook_fn(dataset, seed, cfg)

    # =======================================
    # ========== run main training ==========
    # =======================================
    try:
        metrics, generator_model = main_module.run_single_trial(dataset, eval_hook_fn, cfg)
        if cfg.save_generator:
            results_dir = 'final_model/'
            os.makedirs(results_dir, exist_ok=True)
            generator_model.save_weights(f'{results_dir}/gen_model')
            logging.info(f'Saved model to {os.path.abspath(results_dir)}')
    except Exception as e:
        logging.exception(e)
        return 1e10
    return metrics


if __name__ == '__main__':
    main()
