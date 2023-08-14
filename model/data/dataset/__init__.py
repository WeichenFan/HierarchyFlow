
from .augmentation import *
from .common_dataset import CommonDataset
from .Dehaze_dataset import DehazeDataset
from .hd_dataset import HdDataset
from .de_dataset import DeDataset

def dataset_entry(dataset, **kwargs):
    return globals()[dataset]( **kwargs)


def aug_entry(config):
    aug = globals()[config['type']](**config['kwargs'])
    aug = Compose(aug)
    return aug
