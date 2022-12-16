import collections
import gc
import logging
import math
import time
from datetime import timedelta
from functools import partial

import tensorflow as tf
import tree
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from src.data.abstract_dataset import AbstractDataset
from src.metrics import classification, confusion_matrix, fid
from src.utils.generator_input import create_generator_input_dataset
from src.utils.plot_utils import plot_grid_image

real_model_acc = None
real_model_activations = None

MAX_GENERATED_DATASET_SIZE = 60000


def get_eval_hook_fn(dataset: AbstractDataset,
                     seed,
                     cfg):
    """Returns an eval_hook function to pass to training loop

    Args:
      dataset: An `AbstractDataset` instance holding the tf.data.Datasets.
      seed: Random seed to generate images
      cfg: A DictConfig
    """
    global real_model_acc
    global real_model_activations

    gen_input_dataset = create_generator_input_dataset(cfg.training.batch_size, cfg.dataset.n_classes, cfg.model.latent_dim)

    def get_image(image, _):
        return image

    if dataset.is_federated:
        central_train_dataset = (dataset.train_ds.create_tf_dataset_from_all_clients()
                                 .shuffle(buffer_size=1000, reshuffle_each_iteration=False)
                                 .cache()
                                 .prefetch(AUTOTUNE))
    else:
        central_train_dataset = dataset.train_ds

    num_train_samples = dataset.dataset_size['train']
    if num_train_samples > MAX_GENERATED_DATASET_SIZE:
        num_train_samples = MAX_GENERATED_DATASET_SIZE
        logging.info(f'Setting generated dataset size to {MAX_GENERATED_DATASET_SIZE}, since real size is too large')
    num_train_batches = math.ceil(num_train_samples / cfg.training.batch_size)
    logging.info(f'Generated Dataset will include {num_train_batches} batches ({num_train_batches * cfg.training.batch_size} samples)')

    classification_model = classification.make_classifier_model(cfg.dataset)
    initial_cls_model_weights = classification_model.get_weights()

    # Prepare static eval metrics for real test dataset
    if not cfg.evaluation.real_classifier:
        real_acc = dataset.get_default_accuracy()
        logging.info(f'Skipped real classifier evaluation and set real acc to {real_acc}')
    elif real_model_acc is not None:
        real_acc = real_model_acc
        real_activations = real_model_activations
        logging.info('Skipped re-evaluation of real classifier in second or later --hyperopt trial')
    else:
        if cfg.evaluation.run_cls_eval:
            logging.info('Evaluating acc of real classifier. To skip this use real_classifier = False')
            real_acc, _ = classification.run_classifier(classification_model,
                                                        train_dataset=central_train_dataset,
                                                        test_dataset=dataset.test_ds,
                                                        val_dataset=dataset.val_ds,
                                                        epochs=50, verbose=0)
            real_cls_model_weights = classification_model.get_weights()
            real_model_acc = real_acc

    if cfg.evaluation.run_fid_eval:
        logging.info('evaluating InceptionV3 activations of real DS for FID')
        if real_model_activations is not None:
            real_activations = real_model_activations
        else:
            test_images = dataset.test_ds.map(get_image, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
            upscaled_test_data = test_images.map(
                partial(fid.scale_element, new_shape=[299, 299]), num_parallel_calls=AUTOTUNE).prefetch(
                AUTOTUNE)
            inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
            real_activations = inception_model.predict(upscaled_test_data)
            real_model_activations = real_activations
            # Free memory
            del upscaled_test_data
            del test_images

    # Free memory?
    del central_train_dataset
    gc.collect()

    gen_dataset_input_iterator = iter(gen_input_dataset)

    def eval_hook(generator: tf.keras.Model, epoch_num, additional_eval_fn=None):
        logging.info('Starting evaluation...')

        start_time = time.time()
        metrics = {}

        # Plot images
        predictions = generator(seed, training=False)
        tf.debugging.assert_all_finite(predictions, 'sample images include NaN or Inf')
        plot_grid_image(images=predictions, image_size=cfg.dataset.data_dim,
                        file_name=f'images/round_{epoch_num}-synthetic_images.png')

        # Potentially run additional eval fn
        if additional_eval_fn is not None:
            additional_eval_fn(generator)

        # Generate random data
        def fake_image_generator():
            while True:
                random_noise, labels = next(gen_dataset_input_iterator)
                gen_output = generator([random_noise, labels], training=False)
                yield gen_output, labels

        # Todo maybe use dataset generation like for _create_gen_inputs_dataset
        if any((cfg.evaluation.run_cls_eval, cfg.evaluation.run_fid_eval)):
            generated_dataset = tf.data.Dataset.from_generator(
                fake_image_generator,
                (tf.float32, tf.int64),
                (tf.TensorShape([cfg.training.batch_size, cfg.dataset.data_dim, cfg.dataset.data_dim, cfg.dataset.data_ch]),
                 tf.TensorShape(cfg.training.batch_size)))
            generated_dataset = generated_dataset.take(num_train_batches).prefetch(AUTOTUNE)

            # Run classifier
            if cfg.evaluation.run_cls_eval:
                reset_classification_model()
                metrics['classification'] = run_classification_eval(generated_dataset)

            if cfg.evaluation.run_cm_eval:
                run_cm_eval(generated_dataset, real_cls_model_weights, epoch_num)

            if cfg.evaluation.run_fid_eval:
                # remove labels to prepare for fid calculation
                generated_dataset = generated_dataset.map(get_image, num_parallel_calls=AUTOTUNE)
                metrics['fid'] = run_fid_eval(generated_dataset)

        # Flatten metrics dict
        flat_metrics = tree.flatten_with_path(metrics)
        flat_metrics = [('/'.join(map(str, path)), item) for path, item in flat_metrics]
        flat_metrics = collections.OrderedDict(flat_metrics)

        # Log how long it took to compute/write metrics.
        eval_time = time.time() - start_time
        logging.info(f'Doing evaluation took {str(timedelta(seconds=eval_time))[2:-7]}')

        # Cleanup
        if any((cfg.evaluation.run_cls_eval, cfg.evaluation.run_fid_eval)):
            del generated_dataset
            gc.collect()

        return flat_metrics

    def run_classification_eval(generated_dataset):
        logging.debug('Running classification evaluation')
        start_time = time.time()
        gen_acc, _ = classification.run_classifier(classification_model,
                                                   train_dataset=generated_dataset,
                                                   test_dataset=dataset.test_ds,
                                                   val_dataset=dataset.val_ds,
                                                   epochs=50,
                                                   verbose=0)
        logging.debug(f'classification eval took {time.time() - start_time} sec')

        return {'real_acc': real_acc,
                'gen_acc': gen_acc,
                'acc_diff': real_acc - gen_acc}

    def run_fid_eval(generated_dataset):
        logging.debug('Running FID evaluation')
        start_time = time.time()
        fid_value = fid.calculate_fid(inception_model, generated_dataset, real_activations)
        logging.debug(f'fid eval took {time.time() - start_time} sec')

        return {'fid': fid_value}

    def run_cm_eval(generated_dataset, cls_weights, epoch_num):
        classification_model.set_weights(cls_weights)
        return confusion_matrix.get_cm_image(generated_dataset, classification_model, dataset.class_labels,
                                             file_name=f'images/round_{epoch_num}-conf_mat.png')

    def reset_classification_model():
        classification_model.set_weights(initial_cls_model_weights)

    return eval_hook
