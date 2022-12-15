from functools import partial

import tensorflow_federated as tff
from tensorflow.python.data import AUTOTUNE

from src.data.abstract_dataset import AbstractDataset


class CIFAR100Dataset(AbstractDataset):
    @property
    def class_labels(self):
        return None

    @property
    def dataset_size(self):
        # https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data
        if self.dataset_cfg.use_val_data:
            return {'train': 50000, 'test': 5000, 'val': 5000}
        return {'train': 50000, 'test': 10000}

    @property
    def local_dataset_sizes(self):
        return 100.0, 0.0

    def get_default_accuracy(self):
        return 0.553

    def get_dataset_size_for_client(self, client_id):
        assert self.is_federated[0]
        pass    # todo

    def _load_tf_dataset(self):
        self._load_tff_dataset()
        self.train_ds = self.train_ds.create_tf_dataset_from_all_clients()

    def _load_tff_dataset(self, distribution='equal'):
        cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
        label_key = 'coarse_label' if self.dataset_cfg.n_classes == 20 else 'label'

        def element_fn(element):
            if self.normalisation_mean_zero:
                return element['image'] / 127.5 - 1, element[label_key]
            else:
                return element['image'] / 255, element[label_key]

        def preprocess_federated_dataset(dataset, total_dataset_size, cache=True):
            preprocessed_ds = (dataset.shuffle(buffer_size=round(total_dataset_size / len(cifar_train.client_ids)),
                                               reshuffle_each_iteration=False)
                                      .map(element_fn, num_parallel_calls=AUTOTUNE))
            if cache:
                return preprocessed_ds.cache()
            return preprocessed_ds

        def preprocess_centralized_dataset(dataset, total_dataset_size, cache=True):
            preprocessed_ds = (dataset
                               .shuffle(buffer_size=total_dataset_size,
                                        reshuffle_each_iteration=False)
                               .map(element_fn, num_parallel_calls=AUTOTUNE))
            if cache:
                return preprocessed_ds.cache()
            return preprocessed_ds

        self.train_ds = cifar_train.preprocess(partial(preprocess_federated_dataset,
                                                       total_dataset_size=self.dataset_size['train']))
        if self.is_federated[1]:
            self.test_ds = cifar_test.preprocess(partial(preprocess_federated_dataset,
                                                         total_dataset_size=self.dataset_size['test'],
                                                         cache=(not self.dataset_cfg.use_val_data)))
            if self.dataset_cfg.use_val_data:
                self.val_ds = self.test_ds.preprocess(lambda ds: ds.shard(num_shards=2, index=0).cache())
                self.test_ds = self.test_ds.preprocess(lambda ds: ds.shard(num_shards=2, index=0).cache())
        else:
            self.test_ds = preprocess_centralized_dataset(cifar_test.create_tf_dataset_from_all_clients(),
                                                          total_dataset_size=self.dataset_size['test'],
                                                          cache=(not self.dataset_cfg.use_val_data))
            if self.dataset_cfg.use_val_data:
                self.val_ds = self.test_ds.shard(num_shards=2, index=0).cache()
                self.test_ds = self.test_ds.shard(num_shards=2, index=1).cache()
