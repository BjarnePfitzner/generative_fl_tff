import os
from pathlib import Path

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from src.data.abstract_dataset import AbstractDataset


path = Path(os.getenv('CHEST_XRAY_PATH', '/dhc/dsets/chest_xray_pneumonia_kaggle'))


class CXRDataset(AbstractDataset):
    @property
    def class_labels(self):
        return ['Normal', 'Pneumonia']

    @property
    def dataset_size(self):
        # https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
        if self.dataset_cfg.use_val_data:
            return {'train': 5216, 'test': 640, 'val': 16}
        return {'train': 5216, 'test': 656}

    @property
    def local_dataset_sizes(self):
        return 521.6, 0.48989794855663565

    def get_default_accuracy(self):
        return 0.748

    def get_dataset_size_for_client(self, client_id):
        assert self.is_federated
        return 521

    def _load_tf_dataset(self):
        def read_img(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img,
                                  size=[self.dataset_cfg.data_dim, self.dataset_cfg.data_dim],
                                  method='bilinear', antialias=True)
            if self.normalisation_mean_zero:
                img = (img * 2) - 1

            if tf.strings.split(file_path, os.path.sep)[-2] == 'NORMAL':
                label = 0
            else:
                label = 1

            return img, label

        train_data = tf.data.Dataset.list_files(f'{path}/train/*/*.jpeg')
        if self.dataset_cfg.use_val_data:
            test_data = tf.data.Dataset.list_files(f'{path}/test/*/*.jpeg')
        else:
            test_data = tf.data.Dataset.list_files([f'{path}/test/*/*.jpeg', f'{path}/val/*/*.jpeg'])

        self.train_ds = (train_data
                         .shuffle(buffer_size=self.dataset_size['train'], reshuffle_each_iteration=False)
                         .map(read_img, num_parallel_calls=AUTOTUNE)
                         .cache()
                         )
        self.test_ds = (test_data
                        .shuffle(buffer_size=self.dataset_size['test'], reshuffle_each_iteration=False)
                        .map(read_img, num_parallel_calls=AUTOTUNE)
                        .cache()
                        )

        if self.dataset_cfg.use_val_data:
            val_data = tf.data.Dataset.list_files(f'{path}/val/*/*.jpeg')
            self.val_ds = (val_data
                           .shuffle(buffer_size=self.dataset_size['val'], reshuffle_each_iteration=False)
                           .map(read_img, num_parallel_calls=AUTOTUNE)
                           .cache()
                           )
