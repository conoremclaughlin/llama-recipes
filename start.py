from dataclasses import dataclass
from recipes.quickstart.finetuning.datasets.orv_dataset import get_custom_dataset


@dataclass
class orv_dataset:
    dataset: str = "orv_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/orv_dataset.py"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/content/drive/My Drive/training/orv"


dataset_config = orv_dataset()
dataset_config.data_dir = 'recipes/quickstart/finetuning/datasets/orv'
dataset = get_custom_dataset(dataset_config, None, 'train')

print(f'Our new dataset: {dataset}')
