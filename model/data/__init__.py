
from model.data.GTA.GTA_train import get_mix_GTA_dataset
from model.data.COCO2Wiki.COCO2Wiki_train import get_COCO2Wiki_dataset,get_COCO2Wiki_dataset_HD

def get_dataset_entry(config, **kwargs):
    return globals()[config['dataset']['arch']](config)