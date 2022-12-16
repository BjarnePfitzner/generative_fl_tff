import logging
import math
from abc import ABC, abstractmethod
from copy import copy
from enum import Enum
from functools import reduce

import numpy as np
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow_federated.python.simulation.datasets import ClientData

from src.utils.plot_utils import plot_grid_image


class DatasetType(Enum):
    CXR = "CXR"
    CHEXPERT = "CheXpert"
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    FEMNIST = "FEMNIST"
    CIFAR100 = "CIFAR100"
    FASHION_MNIST = "FashionMNIST"
    CELEBA = "CelebA"

    @classmethod
    def from_value(cls, value):
        try:
            return cls(value)
        except ValueError:
            valid_values = [d.value for d in DatasetType]
            raise ValueError(f"The dataset {value} is not supported. Use one of {', '.join(valid_values)}")


class AbstractDataset(ABC):
    def __init__(self, dataset_cfg, normalisation_mean_zero, is_federated):
        self.dataset_cfg = dataset_cfg
        self.normalisation_mean_zero = normalisation_mean_zero
        self.is_federated = is_federated

        self.train_ds = None
        self.test_ds = None
        self.val_ds = None
        self.is_preprocessed = False

        if is_federated:
            logging.info(f'Loading {self.name} TFF dataset...')
            self._load_tff_dataset()
            self.client_ids = copy(self.train_ds.client_ids)
        else:
            logging.info(f'Loading {self.name} TF dataset...')
            self._load_tf_dataset()
            self.client_ids = ['central_client']
        self.print_dataset_sizes()

    @property
    def name(self):
        return self.dataset_cfg.name

    @property
    def type(self):
        return DatasetType.from_value(self.dataset_cfg.name)

    @property
    @abstractmethod
    def dataset_size(self):
        pass

    @property
    def local_dataset_sizes(self):
        return math.floor(self.dataset_size['train'] / self.dataset_cfg.n_clients), 0.0

    @property
    @abstractmethod
    def class_labels(self):
        pass

    @abstractmethod
    def get_default_accuracy(self):
        pass

    def get_dataset_size_for_client(self, client_id):
        assert self.is_federated
        return self.local_dataset_sizes[0]

    @abstractmethod
    def _load_tf_dataset(self):
        pass

    def _load_tff_dataset(self):
        self._load_tf_dataset()

        if self.dataset_cfg.class_distribution == 'iid':
            self.train_ds = self._create_federated_dataset(self.train_ds, self.dataset_cfg.n_clients)
        elif self.dataset_cfg.class_distribution.endswith('-class-nonIID'):
            self.train_ds = self._prepare_non_iid_data(self.train_ds)
        else:
            raise NotImplementedError('Only iid distribution implemented so far')

    def preprocess_datasets(self, train_batch_size, test_batch_size, local_epochs, local_dp=False):
        if self.is_federated:
            if local_dp:
                self._remove_clients_with_too_little_data(train_batch_size)
            self.train_ds = self._preprocess_tff_dataset(self.train_ds, train_batch_size, local_epochs, drop_remainder=local_dp)
        else:
            self.train_ds = self._preprocess_tf_dataset(self.train_ds, train_batch_size, drop_remainder=local_dp)

        self.test_ds = self._preprocess_tf_dataset(self.test_ds, test_batch_size, drop_remainder=local_dp)
        if self.dataset_cfg.use_val_data:
            self.val_ds = self._preprocess_tf_dataset(self.val_ds, test_batch_size, drop_remainder=local_dp)

        self.is_preprocessed = True

    @classmethod
    def _preprocess_tff_dataset(cls, dataset, batch_size, local_epochs, drop_remainder=False):
        def preprocess_fn(ds):
            return (ds
                    .repeat(local_epochs)
                    .batch(batch_size, drop_remainder=drop_remainder)
                    .prefetch(AUTOTUNE))

        return dataset.preprocess(preprocess_fn)

    @classmethod
    def _preprocess_tf_dataset(cls, dataset, batch_size, drop_remainder=False):
        return (dataset
                .batch(batch_size, drop_remainder=drop_remainder)
                .prefetch(AUTOTUNE))

    @classmethod
    def _create_federated_dataset(cls, dataset, n_clients, sizes=None):
        if sizes is None:
            def create_tf_dataset_for_client_fn(client_id):
                the_id = int(client_id[client_id.find('_') + 1:])
                return dataset.shard(num_shards=n_clients, index=the_id)

            return ClientData.from_clients_and_fn([f'client_{str(the_id)}' for the_id in range(n_clients)],
                                                  create_tf_dataset_for_client_fn)
        else:
            raise NotImplementedError('Predefined client dataset sizes are not supported yet')

    def _create_single_class_datasets(self, dataset):
        single_class_datasets = {}
        for label in range(self.dataset_cfg.n_classes):
            single_class_datasets[label] = dataset.filter(lambda _, y: y == label)
        return single_class_datasets

    def _prepare_non_iid_data(self, dataset):
        n_classes_per_client = int(self.dataset_cfg.class_distribution[0])
        single_class_datasets = self._create_single_class_datasets(dataset)
        n_clients_per_class = int(
            self.dataset_cfg.n_clients * n_classes_per_client / self.dataset_cfg.n_classes)
        while True:
            try:
                client_budget_for_class = {label: n_clients_per_class for label in range(self.dataset_cfg.n_classes)}
                classes_for_client = {}
                for i in range(self.dataset_cfg.n_clients):
                    available_classes = list(client_budget_for_class.keys())
                    selected_classes = np.random.choice(available_classes, n_classes_per_client,
                                                        replace=False,
                                                        p=[client_budget_for_class[c] / sum(
                                                            client_budget_for_class.values())
                                                           for c in client_budget_for_class.keys()])
                    classes_for_client[f'client_{i}'] = []  # holds information in the form (class, shard_id)
                    for selected_class in selected_classes:
                        client_budget_for_class[selected_class] -= 1
                        classes_for_client[f'client_{i}'].append(
                            (selected_class, client_budget_for_class[selected_class]))
                        if client_budget_for_class[selected_class] == 0:
                            del client_budget_for_class[selected_class]
            except ValueError:
                continue
            else:
                break

        def prepare_single_client_data(client_id):
            datasets = []
            for label, shard_id in classes_for_client[client_id]:
                datasets.append(single_class_datasets[label].shard(num_shards=n_clients_per_class,
                                                                   index=shard_id))
            return reduce(lambda x, y: x.concatenate(y), datasets).shuffle(
                buffer_size=int(self.dataset_size['train'] / self.dataset_cfg.n_clients)).cache()

        return ClientData.from_clients_and_fn([f'client_{str(the_id)}' for the_id in range(self.dataset_cfg.n_clients)],
                                              prepare_single_client_data)

    def _remove_clients_with_too_little_data(self, train_batch_size):
        new_client_ids = copy(self.client_ids)
        for client_id in self.client_ids:
            if self.get_dataset_size_for_client(client_id) < train_batch_size:
                new_client_ids.remove(client_id)
        if len(new_client_ids) < len(self.client_ids):
            logging.info(f'{len(new_client_ids)} clients left after removals (from {len(self.client_ids)}).')
            self.client_ids = new_client_ids
            self.dataset_cfg.n_clients = len(new_client_ids)

    def log_sample_data(self):
        if self.is_preprocessed:
            logging.info('Not logging sample data since data has been preprocessed already.')
            return

        logging.info('Logging sample data')
        sample_data = np.array(list(map(lambda x: x[0], self.test_ds.take(16).as_numpy_iterator())))
        plot_grid_image(images=sample_data,
                        image_size=self.dataset_cfg.data_dim,
                        file_name='images/sample_images.png')

    def print_dataset_sizes(self):
        logging.info(f'Dataset sizes: {self.dataset_size}')
        if self.is_federated:
            logging.info(f'Local train sizes: {round(self.local_dataset_sizes[0], 2)} +/- {round(self.local_dataset_sizes[1], 2)}')
