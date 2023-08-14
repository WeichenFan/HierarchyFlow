from model.data.dataset import dataset_entry, aug_entry


def get_COCO2Wiki_dataset(config):
    source_root = config.get('source_root','')
    target_root = config.get('target_root','')
    use_mc = config.get('use_mc',False)

    aug = config['augmentation']
    transform = aug_entry(aug)
    dataset = dataset_entry(
            config['dataset']['name'],
            rootA = source_root,
            rootB = target_root,
            transform = transform,
            use_mc = use_mc)

    return dataset


def get_COCO2Wiki_dataset_HD(config):
    source_root = config.get('source_root','')
    target_root = config.get('target_root','')
    use_mc = config.get('use_mc',False)

    aug = config['augmentation']
    aug_target = config['augmentation_target']

    transform = aug_entry(aug)
    transform_target = aug_entry(aug_target)
    dataset = dataset_entry(
            config['dataset']['name'],
            rootA = source_root,
            rootB = target_root,
            transform = transform,
            transform_target = transform_target,
            use_mc = use_mc)

    return dataset