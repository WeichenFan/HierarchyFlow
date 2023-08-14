import mc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image
import random
import spring.linklink as link
import math
import os
import torch 
import cv2 as cv
import skimage.measure
from torchvision.utils import save_image
from torchvision import transforms
import json

class BaseDataset(Dataset):

    def _init_ceph(self):
        if not self.initialized_ceph:
            # import ceph
            # self.s3_client = ceph.S3Client()
            # self.initialized = True
            from petrel_client.client import Client
            self.s3_client = Client()
            self.initialized_ceph = True

    def _init_memcached(self):
        if not self.initialized_mc:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized_mc = True

    def _common_loader(self):
        def pil_loader(img_str):
            buff = io.BytesIO(img_str)
            with Image.open(buff) as img:
                img = img.convert('RGB')
            return img
        return pil_loader

    def _aug_loader(self):
        def pil_loader(img_str):
            buff = io.BytesIO(img_str)
            with Image.open(buff) as img:
                if random.random() < 0.5:
                    img = img.convert('L').convert('RGB')
                else:
                    img = img.convert('RGB')
            return img
        return pil_loader

    def _random_overlap(self,img):
        if random.random() < 0.4:
            sl = 0.02
            sh = 0.4
            r1 = 0.3
            means = [125, 122, 113]
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = means[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = means[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = means[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = means[0]
        return img

    def _get_img_from_mask(self,img_path, mask_path):
        mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask[mask!=0] = 1
        lbl = skimage.measure.label(mask)
        props = skimage.measure.regionprops(lbl)
        bboxes = []
        for prop in props:
            bboxes.append([prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]])
        if len(bboxes) > 1:
            min_x = 100000
            min_y = 100000
            max_x = 0
            max_y = 0
            for item in bboxes:
                if item[0] < min_x: min_x = item[0]
                if item[1] < min_y: min_y = item[1]
                if item[2] > max_x: max_x = item[2]
                if item[3] > max_y: max_y = item[3]
            bbox = [min_x, min_y, max_x, max_y]
            # bbox = [min(bboxes[:, 0]), min(bboxes[:, 1]), max(bboxes[:, 2]), max(bboxes[:, 3])]
        else:
            bbox = bboxes[0]
        img_cut = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask_cut = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return img_cut[:,:,::-1], mask_cut

    def _random_background(self,img, prob, extend_rate, cut_rate, backgrounds, mask=None):
        background = random.choice(backgrounds).copy()
        if random.random() < prob:
            extend_ratio = [random.random()*extend_rate, random.random()*extend_rate]
            cut_ratio = [random.random()*cut_rate, random.random()*cut_rate]
            # print(background.shape)
            h, w, _ = img.shape
            if random.randint(0, 1):
                back_h = int(h*extend_ratio[0])
                back_w = int(w*extend_ratio[1])
                new_h = back_h + h
                new_w = back_w + w
            else:
                back_h = int(h*cut_ratio[0])
                back_w = int(w*cut_ratio[1])
                new_h = h
                new_w = w
                cut_h = random.randint(0, back_h)
                cut_w = random.randint(0, back_w)
                img = img[cut_h:h-back_h+cut_h, cut_w:w-back_w+cut_w, :]
                if mask is not None:
                    mask = mask[cut_h:h-back_h+cut_h, cut_w:w-back_w+cut_w]
                h, w, _ = img.shape
            if background.shape[0] < new_h or background.shape[1] < new_w:
                print('image shape({}) too large for background({})'.format(img.shape, background.shape))
                return img
            start_h = random.randint(0, background.shape[0] - new_h)
            start_w = random.randint(0, background.shape[1] - new_w)
            img_out = background[start_h:start_h+new_h, start_w:start_w+new_w, :]
            start_h = random.randint(0, back_h)
            start_w = random.randint(0, back_w)
            if mask is None:
                img_out[start_h:start_h+h, start_w:start_w+w, :] = img
            else:
                img_out[start_h:start_h+h, start_w:start_w+w, :][mask!=0] = img[mask!=0]
            
            if random.randint(0, 1):
                ksize=random.randint(0, 3)*2+1
                img_out = cv.GaussianBlur(img_out, (ksize, ksize), cv.BORDER_DEFAULT)
            return img_out
        else:
            h, w, _ = img.shape
            if background.shape[0] < h or background.shape[1] < w:
                img_out = cv.resize(background, (w, h))
            else:
                start_h = random.randint(0, background.shape[0] - h)
                start_w = random.randint(0, background.shape[1] - w)
                img_out = background[start_h:start_h+h, start_w:start_w+w, :]
            if mask is None:
                img_out[:h, :w, :] = img
            else:
                img_out[mask!=0] = img
            
            if random.randint(0, 1):
                ksize=random.randint(0, 3)*2+1
                img_out = cv.GaussianBlur(img_out, (ksize, ksize), cv.BORDER_DEFAULT)
            return img_out
