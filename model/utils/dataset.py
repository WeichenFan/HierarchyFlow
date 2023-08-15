from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_img_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        raise Exception('{} not found'.format(path))
    else:
        return img

class BaseDataset(data.Dataset):
    def __init__(self, cfg):
        src_list = cfg.source_list
        tgt_list = cfg.target_list
        self.src_list = []
        self.tgt_list = []
        with open(src_list, 'r') as f:
            for line in f.readlines():
                self.src_list.append(line.strip())
        with open(tgt_list, 'r') as f:
            for line in f.readlines():
                self.tgt_list.append(line.strip())
        self.src_len = len(self.src_list)
        self.tgt_len = len(self.tgt_list)

        transform_options = cfg.transform
        height, width = cfg.height, cfg.width
        transform_list = []
        transform_list.append(transforms.Resize((height, width)))
        if 'crop' in transform_options:
            transform_list.append(transforms.RandomCrop(height))
        if 'h_flip' in transform_options:
            transform_list.append(transforms.RandomHorizontalFlip(0.5))
        if 'v_flip' in transform_options:
            transform_list.append(transforms.RandomVerticalFlip(0.5))
        transform_list.append(transforms.ToTensor())
        if 'normalize' in transform_options:
            transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transform = transforms.Compose(transform_list)
        self.image_loader = pil_img_loader
        self.random_pair = cfg.random_pair
        self.return_name = cfg.return_name

    def __getitem__(self, index):
        if self.random_pair:
            tgt_idx = random.randint(0, self.tgt_len-1)
        else:
            tgt_idx = index % self.tgt_len
        src_img = pil_img_loader(self.src_list[index])
        tgt_img = pil_img_loader(self.tgt_list[tgt_idx])
        src_img = self.transform(src_img)
        tgt_img = self.transform(tgt_img)
        
        if not self.return_name:
            return src_img, tgt_img
        else:
            return src_img, tgt_img, '{}_to_{}.png'.format(self.src_list[index].split('/')[-1].split('.')[0], self.tgt_list[tgt_idx].split('/')[-1].split('.')[0])

    def __len__(self):
        return self.src_len


def get_dataset(cfg):
    dataset = BaseDataset(cfg)
    return dataset