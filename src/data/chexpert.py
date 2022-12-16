import logging
import os
from pathlib import Path

from hydra.utils import to_absolute_path
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from src.data.abstract_dataset import AbstractDataset

NEW_CHEXPERT_INFO_PATH = to_absolute_path('./src/data/chexpert_info.csv')
path = Path('/dhc/dsets/ChestXrays/CheXpert/CheXpert-v1.0-small')
#path = Path('/mnt/dsets/chexpert/CheXpert-v1.0-small')


class CheXpertDataset(AbstractDataset):
    @property
    def name(self):
        return 'CheXpert'

    @property
    def class_labels(self):
        return ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                'Fracture']

    @property
    def dataset_size(self):
        total_size = 35144
        train_size = round(total_size * (1 - self.dataset_cfg.test_fraction))
        if self.dataset_cfg.use_val_data:
            test_size = round(total_size * self.dataset_cfg.test_fraction / 2)
            val_size = total_size - (train_size + test_size)
            return {'train': train_size, 'test': test_size, 'val': val_size}
        else:
            test_size = round(total_size * self.dataset_cfg.test_fraction)
            return {'train': train_size, 'test': test_size}

    @property
    def local_dataset_sizes(self):
        return 1405.75, 0.4330127018922193

    def get_default_accuracy(self):
        return 0.342

    def get_dataset_size_for_client(self, client_id):
        assert self.is_federated
        return 1406

    def _load_tf_dataset(self):

        def read_img(entry):
            img = tf.io.read_file(str(path) + entry['Path'])
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            h, w = tf.shape(img)[0], tf.shape(img)[1]
            if h > w:
                img = tf.image.crop_to_bounding_box(img, (h - w) // 2, 0, w, w)
            else:
                img = tf.image.crop_to_bounding_box(img, 0, (w - h) // 2, h, h)
            img = tf.image.resize(img,
                                  size=[self.dataset_cfg.data_dim, self.dataset_cfg.data_dim],
                                  method='bilinear', antialias=True)
            if self.normalisation_mean_zero:
                img = (img * 2) - 1

            return img, entry['Int Label']

        if not os.path.exists(NEW_CHEXPERT_INFO_PATH):
            logging.info('parsing chexpert info...')
            # Parse CheXpert info files and save information of files with only one (or no) finding
            train_info = pd.read_csv(f'{path}/train.csv')
            valid_info = pd.read_csv(f'{path}/valid.csv')
            all_info = pd.concat([train_info, valid_info], axis=0)

            # Remove first path component
            all_info['Path'] = all_info['Path'].apply(lambda path: path[path.find('/'):])

            # Remove lateral view and support devices
            all_info = all_info[all_info['Frontal/Lateral'] == 'Frontal']
            all_info = all_info[all_info['Support Devices'] != 1.0]

            # Remove unused information
            all_info.drop(columns=['Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 'Support Devices'], inplace=True)

            # Replace uncertainty label with negative label
            all_info.replace(-1.0, 0.0, inplace=True)

            # Replace nans with zeros
            all_info.fillna(0.0, inplace=True)

            # Remove entries with more than one finding
            all_info = all_info[all_info.sum(axis=1, numeric_only=True) == 1.0]

            # Add no finding to entries with no finding
            all_info['No Finding'].iloc[all_info.sum(axis=1, numeric_only=True) == 0.0] = 1.0

            label_encoding_dict = {}
            counter = 0
            for col_name in all_info.drop(columns=['Path']).columns:
                label_encoding_dict[col_name] = counter
                counter += 1

            # Restructure df to have single label column
            chexpert_info = pd.DataFrame(columns=['Path', 'Label', 'Int Label'])
            chexpert_info['Label'] = all_info.drop(columns=['Path']).idxmax(axis=1)
            # Integer encoding
            chexpert_info['Int Label'] = chexpert_info['Label'].replace(label_encoding_dict)
            chexpert_info['Path'] = all_info['Path']

            chexpert_info.to_csv(NEW_CHEXPERT_INFO_PATH, index=False)
            chexpert_info = dict(chexpert_info)
        else:
            logging.info('loading chexpert info from file...')
            #chexpert_info = []
            #with open(NEW_CHEXPERT_INFO_PATH, 'r') as info_file:
            #    logging.info('opened file')
            #    reader = csv.DictReader(info_file)
            #    logging.info('built reader')
            #    for row in reader:
            #        logging.info(row)
            #        chexpert_info.append(row)
            chexpert_info = dict(pd.read_csv(NEW_CHEXPERT_INFO_PATH, engine='c'))

        all_data = (tf.data.Dataset.from_tensor_slices(chexpert_info)
                    .shuffle(buffer_size=len(chexpert_info), reshuffle_each_iteration=False)
                    .map(read_img, num_parallel_calls=AUTOTUNE)
                    )

        train_data_size = self.dataset_size['train']
        test_data_size = self.dataset_size['test']
        self.train_ds = all_data.take(train_data_size).cache()
        self.test_ds = all_data.skip(train_data_size).take(test_data_size).cache()
        if self.dataset_cfg.use_val_data:
            self.val_ds = all_data.skip(train_data_size + test_data_size).cache()
