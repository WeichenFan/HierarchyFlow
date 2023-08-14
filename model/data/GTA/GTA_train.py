from model.data.dataset import dataset_entry, aug_entry


def get_mix_GTA_dataset(config):
    source_root = config.get('source_root','')
    target_root = config.get('target_root','')

    aug = config['augmentation']
    transform = aug_entry(aug)
    dataset = dataset_entry(
            config['dataset']['name'],
            rootA = source_root,
            rootB = target_root,
            transform = transform)

    return dataset