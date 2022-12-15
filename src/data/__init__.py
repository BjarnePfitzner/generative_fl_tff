from src.data import femnist, mnist, chexpert, cifar100, cxr, fashion_mnist, celeba
from src.data.abstract_dataset import DatasetType, AbstractDataset


def get_dataset_instance(cfg) -> AbstractDataset:
    is_federated = (cfg.model.type != 'centralized')
    dataset_type = DatasetType.from_value(cfg.dataset.name)

    if dataset_type == DatasetType.MNIST:
        dataset_class = mnist.MNISTDataset
    elif dataset_type in [DatasetType.FMNIST, DatasetType.FEMNIST]:
        dataset_class = femnist.FEMNISTDataset
    elif dataset_type == DatasetType.CIFAR100:
        dataset_class = cifar100.CIFAR100Dataset
    elif dataset_type == DatasetType.CHEXPERT:
        dataset_class = chexpert.CheXpertDataset
    elif dataset_type == DatasetType.CXR:
        dataset_class = cxr.CXRDataset
    elif dataset_type == DatasetType.FASHION_MNIST:
        dataset_class = fashion_mnist.FashionMNISTDataset
    elif dataset_type == DatasetType.CELEBA:
        dataset_class = celeba.CelebADataset
    else:
        raise NotImplementedError('Dataset type not defined')

    dataset = dataset_class(cfg.dataset, cfg.model.normalisation_mean_zero, is_federated)

    if cfg.log_sample_data:
        dataset.log_sample_data(cfg.run_dir)
    return dataset
