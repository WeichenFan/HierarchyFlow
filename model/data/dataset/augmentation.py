import random
import numpy as np
import math
import torchvision.transforms as transforms
import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import PIL
import collections
from torchvision.transforms import ColorJitter,ToTensor,Resize,Normalize,RandomVerticalFlip,RandomHorizontalFlip
import inspect

def eval_aug(input_size_h,input_size_w):
    aug = []
    aug.append([Resize((input_size_h,input_size_w)),Resize((input_size_h,input_size_w))])
    aug.append([ToTensor(),ToTensor()])
    return aug

def De_aug(input_size_h,input_size_w,scale_down):
    aug = []
    aug.append([ToTensor(),ToTensor()])
    return aug

def ROBI_aug(input_size_h,input_size_w,scale_down):
    aug = []
    #aug.append([transforms.RandomResizedCrop((input_size_h,input_size_w),scale=(scale_down, 1.0)),transforms.RandomResizedCrop((input_size_h,input_size_w),scale=(scale_down, 1.0))])
    #aug.append([RandomHorizontalFlip(),RandomHorizontalFlip()])
    #aug.append([RandomVerticalFlip(),RandomVerticalFlip()])
    aug.append([ToTensor(),ToTensor()])
    return aug

def Common_aug(input_size_h,input_size_w,scale_down):
    aug = []
    aug.append([transforms.RandomResizedCrop((input_size_h,input_size_w),scale=(scale_down, 1.0)),transforms.RandomResizedCrop((input_size_h,input_size_w),scale=(scale_down, 1.0))])
    aug.append([RandomHorizontalFlip(),RandomHorizontalFlip()])
    aug.append([RandomVerticalFlip(),RandomVerticalFlip()])
    aug.append([ToTensor(),ToTensor()])
    return aug

def Pair_aug(input_size_h,input_size_w,flip_p,scale_down):
    aug = []
    rr_gen = Random_Resize_Generator(scale=(scale_down, 1.0), ratio=(3. / 4., 4. / 3.))
    r_gen = RandomGenerator()
    aug.append(rr_gen)
    aug.append(r_gen)
    aug.append([Given_Random_Resized_Crop(rr_gen,(input_size_h,input_size_w)),Given_Random_Resized_Crop(rr_gen,(input_size_h,input_size_w))])
    aug.append([Mannual_H_Flip(flip_p,r_gen),Mannual_H_Flip(flip_p,r_gen)])
    aug.append([ToTensor(),ToTensor()])
    return aug

def sr_aug(input_size_h,input_size_w,flip_p,scale_down,mixup_prob):
    aug = []
    rr_gen = Random_Resize_Generator(scale=(scale_down, 1.0), ratio=(3. / 4., 4. / 3.))
    r_gen = RandomGenerator()
    r_gen_vflip = RandomGenerator()
    aug.append(rr_gen)
    aug.append(r_gen)
    aug.append(r_gen_vflip)
    aug.append([Given_Random_Resized_Crop(rr_gen,(input_size_h,input_size_w)),Given_Random_Resized_Crop(rr_gen,(input_size_h,input_size_w))])
    aug.append([Mannual_H_Flip(flip_p,r_gen),Mannual_H_Flip(flip_p,r_gen)])
    aug.append([Mannual_V_Flip(flip_p,r_gen_vflip),Mannual_V_Flip(flip_p,r_gen_vflip)])
    aug.append([Resize((64,64)),])
    aug.append([Resize((512,512)),])
    
    aug.append([ToTensor(),ToTensor()])
    aug.append(Mixup(mixup_prob))

    return aug

def Dehaze_aug(input_size_h,input_size_w,flip_p,scale_down,mixup_prob):
    aug = []
    rr_gen = Random_Resize_Generator(scale=(scale_down, 1.0), ratio=(3. / 4., 4. / 3.))
    r_gen = RandomGenerator()
    r_gen_vflip = RandomGenerator()
    aug.append(rr_gen)
    aug.append(r_gen)
    aug.append(r_gen_vflip)
    aug.append([Given_Random_Resized_Crop(rr_gen,(input_size_h,input_size_w)),Given_Random_Resized_Crop(rr_gen,(input_size_h,input_size_w))])
    aug.append([Mannual_H_Flip(flip_p,r_gen),Mannual_H_Flip(flip_p,r_gen)])
    aug.append([Mannual_V_Flip(flip_p,r_gen_vflip),Mannual_V_Flip(flip_p,r_gen_vflip)])
    
    aug.append([ToTensor(),ToTensor()])
    aug.append(Mixup(mixup_prob))

    return aug

class Mixup(object):
    def __init__(self,prob):
        self.p = prob
    def __call__(self,img):
        if random.uniform(0, 1) > self.p:
            src = img[0]
            tgt = img[1]
            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            src = lam * src + (1 - lam) * tgt
            return [src,tgt]
        else:
            return img

class Given_Random_Resized_Crop(object):
    def __init__(self,gen,size):
        self._gen = gen
        self.size = size
    def __call__(self,img):
        mrrc = Mannual_RRC()
        param_list = self._gen.crop_param
        img = mrrc(img,param_list,self.size)
        return img

class Mannual_RRC(object):
    def __call__(self,img,param,size):
        return F.resized_crop(img, param[0], param[1], param[2], param[3], size, Image.BILINEAR)

class Random_Resize_Generator(object):
    def __init__(self,scale,ratio):
        self.scale = scale
        self.ratio = ratio
    def __call__(self,img):
        width, height = F._get_image_size(img[0])
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            log_ratio = torch.log(torch.tensor(self.ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                self.crop_param = [i,j,h,w]
                return img

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        self.crop_param = [i,j,h,w]
        return img

class RandomGenerator(object):
    def __call__(self, img):
        self.seed = random.uniform(0, 1)

        return img

class Mannual_H_Flip(object):
    def __init__(self, p=0.5, gen=None):
        self.p = p
        self._gen = gen

    def __call__(self, img):

        if self._gen.seed < self.p:
            return F.hflip(img)
        return img

class Mannual_V_Flip(object):
    def __init__(self, p=0.5, gen=None):
        self.p = p
        self._gen = gen

    def __call__(self, img):

        if self._gen.seed < self.p:
            return F.vflip(img)
        return img

def combine_trans_train_cash(**kwargs):
    aug = []
    hd_gen = HandRotate()
    size_gen = GetImgSize()
    aug.append(hd_gen)
    aug.append(size_gen)
    Combine_trans_args = list(inspect.signature(Combine_trans).parameters)
    Combine_trans_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in Combine_trans_args}
    target_angle_rotate_args = list(inspect.signature(target_angle_rotate).parameters)
    target_angle_rotate_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in target_angle_rotate_args}
    aug.append([Combine_trans(hand_gen=hd_gen,**Combine_trans_dict),target_angle_rotate(gen=hd_gen,get_size=size_gen,**target_angle_rotate_dict)])
    return aug

def combine_trans_eval_cash(**kwargs):
    aug = []
    aug.append([eval_angle_img(),eval_angle_trans()])
    return aug

def combine_trans_train_marker(**kwargs):
    aug = []

    Combine_trans_args = list(inspect.signature(Combine_trans).parameters)
    Combine_trans_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in Combine_trans_args}
    aug.append([Combine_trans(**Combine_trans_dict),])
    return aug

class ImageNetPolicy(object):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.range = {
            "shearX": np.linspace(-0.1, 0.1, 10),
            "shearY": np.linspace(-0.1, 0.1, 10)
            #"rotate": np.linspace(-30, 30, 10),
            #"color": np.linspace(0.6, 1.5, 10),
            #"posterize": np.round(np.linspace(4, 8, 10), 0).astype(np.int),
            #"contrast": np.linspace(0.6, 1.5, 10),
            #"sharpness": np.linspace(0.1, 1.9, 10),
            #"brightness": np.linspace(0.5, 1.4, 10),
            #"autocontrast": [0] * 10,
            #"equalize": [0] * 10,
            #"blur": np.linspace(0.5, 1.0, 10),
            #"detail": [0] * 10
        }

    def func(self, op, img, magnitude):
        if op== "shearX": return shear(img, magnitude * 180, direction="x")
        elif op== "shearY": return shear(img, magnitude * 180, direction="y")
        #elif op== "rotate": return img.rotate(magnitude)
        elif op=="color": return ImageEnhance.Color(img).enhance(magnitude)
        elif op=="posterize": return ImageOps.posterize(img, magnitude)
        elif op=="contrast": return ImageEnhance.Contrast(img).enhance(magnitude)
        elif op=="sharpness": return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif op=="brightness": return ImageEnhance.Brightness(img).enhance(magnitude)
        elif op=="autocontrast": return ImageOps.autocontrast(img)
        elif op=="equalize": return ImageOps.equalize(img)
        elif op=="blur": return img.filter(ImageFilter.GaussianBlur(radius=magnitude))
        elif op=="detail": return img.filter(ImageFilter.DETAIL)
        else: print('error ops')

    def __call__(self, img):
        # if random.random() < self.keep_prob:
        #     return img
        # else:
        rand = np.random.randint(0, 10, 2)
        policies = random.sample(list(self.range.keys()), 2)
        if random.random() < 0.5:
            img = self.func(policies[0], img, self.range[policies[0]][rand[0]])
            img = self.func(policies[1], img, self.range[policies[1]][rand[1]])
        return img


def shear(img, angle_to_shear, direction="x"):
    width, height = img.size
    phi = math.tan(math.radians(angle_to_shear))

    if direction=="x":
        shift_in_pixels = phi * height

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

        img = img.transform((int(round(width + shift_in_pixels)), height),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC, fillcolor=(0, 0, 0))

        return img.resize((width, height), resample=Image.BICUBIC)

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)

        image = img.transform((width, int(round(height + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC, fillcolor=(0, 0, 0))

        return image.resize((width, height), resample=Image.BICUBIC)

def get_params(img, scale, ratio):
    area = img.size[0] * img.size[1]

    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5 and min(ratio) <= (h / (w+1e6)) <= max(ratio):
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            i = random.randint(0, img.size[1] - h)
            j = random.randint(0, img.size[0] - w)
            return i, j, h, w

    # Fallback
    w = min(img.size[0], img.size[1])
    i = (img.size[1] - w) // 2
    j = (img.size[0] - w) // 2
    return i, j, w, w

def scale_chips(img, method=Image.BICUBIC,scale=(0.5,1.0),ratio=(3. / 4., 4. / 3.),prob=0.5,input_size=224):
    if random.random() < prob:
        #print('crop')
        width, height = img.size
        cw = random.randint(width//2,width)
        ch = random.randint(height//2,height)
        T = transforms.RandomCrop((ch,cw))
        img = T(img)
    # i, j, h, w = get_params(img, (0.5,1.0), (3. / 4., 4. / 3.))
    # img = F.crop(img, i, j, h, w)
    ho ,wo =input_size, input_size
    width, height = img.size
    if width == 0 or height == 0:
        print('width: ',width,' height: ',height)
        return img.resize((wo,ho),method)
    

    ratio = min(ho/height,wo/width)
    output_w = int(round(ratio*width))
    output_h = int(round(ratio*height))
    if output_w == 0:
        output_w = min(input_size,width)
        output_h = input_size
    if output_h == 0:
        output_h = min(input_size,height)
        output_w = input_size
    
    output = np.zeros((ho, wo, 3), dtype=np.uint8) + 0
    jitter_h = random.randint(0,ho-output_h)
    jitter_w = random.randint(0,wo-output_w)

    output[jitter_h:output_h+jitter_h, jitter_w:output_w+jitter_w, :] = np.array(img.resize((output_w, output_h), method),dtype=np.uint8)
    return Image.fromarray(output).resize((wo,ho),method)

def scale_chips_test(img, method=Image.BICUBIC,input_size=224):

    # i, j, h, w = get_params(img, (0.5,1.0), (3. / 4., 4. / 3.))
    # img = F.crop(img, i, j, h, w)

    width, height = img.size
    ho ,wo =input_size, input_size

    ratio = min(ho/height,wo/width)
    output_w = int(round(ratio*width))
    output_h = int(round(ratio*height))

    output = np.zeros((ho, wo, 3), dtype=np.uint8)# + 127
    output[:output_h, :output_w, :] = np.array(img.resize((output_w, output_h), method),dtype=np.uint8)
    return Image.fromarray(output).resize((wo,ho),method)

class KeepRatioResize(object):
    def __init__(self, scale=(0.5,1.0), ratio=(3. / 4., 4. / 3.), prob=0.5, input_size=224):
        self.scale = scale
        self.ratio = ratio
        self.prob = prob
        self.input_size = input_size

    def __call__(self, tensor):
        tensor = scale_chips(tensor,scale=self.scale,ratio=self.ratio,prob=self.prob,input_size=self.input_size)
        return tensor


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor



class Add_Salt_Peper_Noise(object):
    def __init__(self,keep_prob=0.5,prob=0.01):
        self.keep_prob = keep_prob
        self.prob = prob

    def add_noise(self,image):
        img = np.array(image)
        h, w, d = img.shape
        n = int(h*w*self.prob)
        for i in range(n):
            if random.randint(0, 1):
                img[random.randint(0,h-1), random.randint(0,w-1), :] = [255, 255, 255]
            else:
                img[random.randint(0,h-1), random.randint(0,w-1), :] = [0, 0, 0]
        kernel_size = 2*random.randint(1, 3)+1
        img = cv2.GaussianBlur(img, ksize=(kernel_size, kernel_size), sigmaX=0, sigmaY=0)
        return img

    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img
        else:
            img = self.add_noise(img)
            img = Image.fromarray(img)
            return img

class LightAug(object):
    def __init__(self, keep_prob=0.5,w=224,h=224,strength=10):
        self.keep_prob = keep_prob
        self.w = w
        self.h = h
        self.strength = strength

    def makeGaussian(self, size, fwhm = 3, center=None):
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        m = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        return m

    def add_mask(self,img,mask):
        img_ = np.array(img)
        mask = cv2.resize(mask, (img_.shape[1],img_.shape[0]), interpolation=cv2.INTER_CUBIC)
        scr_arr = np.array(img_ + mask)#,dtype=np.uint8)


        scr_arr = np.clip(scr_arr,0,255)
        scr_arr = np.array(scr_arr,dtype=np.uint8)

        scr_arr = Image.fromarray(scr_arr)#.convert("RGB")
        return scr_arr

    def get_mask(self,src,w=224,h=224):
        mask = np.zeros((h,w,3))
        mask1 = self.makeGaussian(w,fwhm=random.randint(1,w),center=[random.randint(1,w),random.randint(1,w)])

        mask[:,:,0] = mask[:,:,0] + random.randint(-self.strength,self.strength)*mask1
        mask[:,:,1] = mask[:,:,1] + random.randint(-self.strength,self.strength)*mask1
        mask[:,:,2] = mask[:,:,2] + random.randint(-self.strength,self.strength)*mask1
        return mask

    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img
        else:
            mask_num = random.randint(1,3)
            mask_t = self.get_mask(img,224)
            for i in range(1,mask_num):
                mask_ = self.get_mask(img,224)
                mask_t = mask_t + mask_
            #mask_t = np.resize(mask_t,(self.h,self.w))
            img = self.add_mask(img,mask_t)
            return img

class MotionBlur(object):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def motion_blur(self, image, degree=4, angle=30):
        image = np.array(image)
    
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img
        else:
            img = self.motion_blur(img, random.randint(2, 5), random.randint(0, 360))
            #img = self.motion_blur(img, random.randint(2, 6), random.randint(20, 60))
            img = Image.fromarray(img)
            return img

class VerticalFlip(object):
    def __init__(self, p=0.5, gen=None):
        self.p = p
        self._gen = gen

    def __call__(self, img):

        if self._gen.seed < self.p:
            return F.vflip(img)
        return img

class VerticalFlip_inverse_target(object):
    def __init__(self, p=0.5, gen=None):
        self.p = p
        self._gen = gen

    def __call__(self, target):
        # if isinstance(target,list):
        #     if len(target) > 1:
        #         if self._gen.seed < self.p:
        #             target[0] = 1-target[0]
        #         return target
        if self._gen.seed < self.p:
            if isinstance(target,list):
                return [1-target[0],target[1]]
            return 1-target
        return target

class eval_angle_trans(object):
    def __call__(self, target):
        res = target[:-1]
        ang_ = target[-1]/180.*np.pi
        ang_trans = [(np.sin(ang_)),(np.cos(ang_))]
        res += ang_trans
        return res
        #return [target[0],ang_trans[0],ang_trans[1]]

class eval_angle_img(object):
    def __call__(self, img):
        return img

class target_angle_rotate(object):
    def __init__(self, gen=None,get_size=None,ifcash=False):
        self._gen = gen
        self._get_size = get_size
        self.ifcash = ifcash

    def __call__(self, target):
        if self._gen is not None:
            width,height = self._get_size.seed
            res = target[:-1]
            angle = target[-1]

            if angle > 0:
                sim_dx = 1
                tan_value = np.tan(angle*np.pi/180.)
                sim_dy = sim_dx / tan_value

                sim_dx = width/224. * sim_dx
                sim_dy = height/224. * sim_dy
                b = math.atan2(sim_dx,sim_dy)
                angle = b/np.pi*180
            elif angle < 0:
                sim_dx = -1
                tan_value = np.tan(angle*np.pi/180.)
                sim_dy = sim_dx / tan_value

                sim_dx = width/224. * sim_dx
                sim_dy = height/224. * sim_dy
                b = math.atan2(sim_dx,sim_dy)
                angle = b/np.pi*180

            angle += self._gen.seed
            if angle > 180:
                angle = angle - 360
            elif angle < -180:
                angle = 360 + angle 
            if angle == -180:
                angle = 180
            
            if angle > 0:
                sim_dx = 1
                tan_value = np.tan(angle*np.pi/180.)
                sim_dy = sim_dx / tan_value

                sim_dx = 224./width * sim_dx
                sim_dy = 224./height * sim_dy
                b = math.atan2(sim_dx,sim_dy)
                angle = b/np.pi*180
            elif angle < 0:
                sim_dx = -1
                tan_value = np.tan(angle*np.pi/180.)
                sim_dy = sim_dx / tan_value

                sim_dx = 224./width * sim_dx
                sim_dy = 224./height * sim_dy
                b = math.atan2(sim_dx,sim_dy)
                angle = b/np.pi*180

            if self.ifcash:
                angle = np.abs(angle)
                if angle > 90:
                    angle = 180-angle

            ang_ = angle/180.*np.pi
            ang_trans = [np.sin(ang_),np.cos(ang_)]
            res += ang_trans
            return res
        else:
            res = target[:-1]
            angle = target[-1]

            if angle > 0:
                sim_dx = 1
                tan_value = np.tan(angle*np.pi/180.)
                sim_dy = sim_dx / tan_value

                sim_dx = 224./width * sim_dx
                sim_dy = 224./height * sim_dy
                b = math.atan2(sim_dx,sim_dy)
                angle = b/np.pi*180
            elif angle < 0:
                sim_dx = -1
                tan_value = np.tan(angle*np.pi/180.)
                sim_dy = sim_dx / tan_value

                sim_dx = 224./width * sim_dx
                sim_dy = 224./height * sim_dy
                b = math.atan2(sim_dx,sim_dy)
                angle = b/np.pi*180

            ang_ = angle/180.*np.pi
            ang_trans = [np.sin(ang_),np.cos(ang_)]
            res += ang_trans
            return res
            #return [target[0],ang_trans[0],ang_trans[1]]

class HandRotate(object):
    def __init__(self, degree1=-180, degree2=180):
        self.degree1 = degree1
        self.degree2 = degree2

    def __call__(self, img):
        self.seed = random.uniform(self.degree1, self.degree2)

        return img

class GetImgSize(object):
    def __init__(self, degree1=-180, degree2=180):
        self.degree1 = degree1
        self.degree2 = degree2

    def __call__(self, img):
        self.seed = img[0].size

        return img

class ChooseRotate(object):
    def __init__(self, degree1, degree2, resample=PIL.Image.BILINEAR, expand=False, center=None, fill=(128, 128, 128)):
        self.degree1 = degree1
        self.degree2 = degree2
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def get_params(self,degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        w,h = img.size
        if h > w:
            angle1 = self.get_params(self.degree1)
            return F.rotate(img, angle1, self.resample, self.expand, self.center)
        else:
            angle2 = self.get_params(self.degree2)
            return F.rotate(img, angle2, self.resample, self.expand, self.center)

class CommonRotate(object):
    def __init__(self, degree1, resample=PIL.Image.BILINEAR, expand=False, center=None, fill=(128, 128, 128)):
        self.degree1 = degree1
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def get_params(self,degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        #w,h = img.size
        angle1 = self.get_params(self.degree1)
        return F.rotate(img, angle1, self.resample, self.expand, self.center)

def _iterate_transforms(transforms, x):
    if isinstance(transforms, collections.Iterable):
        for i, transform in enumerate(transforms):
            x[i] = _iterate_transforms(transform, x[i])
    else:
        x = transforms(x)
    return x

class multi_scale_resize(object):
    def __init__(self,input_size):
        self.input_size = input_size
    def __call__(self, img):
        img = scale_chips_test(img,input_size=self.input_size)
        return img

class Combine_trans(object):
    def __init__(self,input_size=None,input_size_h=None,input_size_w=None,rc_proob=0.7,imgnet=False,choose_rotate=True,angle=180,scale_down=0.5,scale_resize=True,padding=True,center_crop=True,keep_ratio_crop_prob=0.5,hand_gen=None,random_overlap=True):
        self.imgnetp = ImageNetPolicy()
        if choose_rotate:
            self.rotate = ChooseRotate([-30,30],[-5,5])
        else:
            self.rotate = CommonRotate([-angle,angle])
        if scale_resize:
            if input_size == None:
                self.resize_crop = transforms.RandomResizedCrop((input_size_h,input_size_w),scale=(scale_down, 1.0))
            else:
                self.resize_crop = transforms.RandomResizedCrop((input_size,input_size),scale=(scale_down, 1.0))
        else:
            self.resize_crop = KeepRatioResize(prob=keep_ratio_crop_prob,input_size=input_size)
        self.scale_resize = scale_resize
        self.padding = padding
        self.center_crop = center_crop
        if input_size == None:
            self.resize = transforms.Resize((input_size_h,input_size_w))
        else:
            self.resize = transforms.Resize((input_size,input_size))
        self.rc_proob = rc_proob
        self.imgnet=imgnet
        self.hand_gen = hand_gen
        self.do_random_overlap = random_overlap

    def random_overlap(self,img):
        if random.random() < 0.4:
            img = np.array(img,dtype=np.uint8)
            sl = 0.02
            sh = 0.2
            r1 = 0.3
            means = [0, 0, 0]
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
            img = Image.fromarray(img)
        return img

    def __call__(self, img):
        w,h = img.size
        #print('ow: ',w)
        #print('oh: ',h)
        aug_tmp = [transforms.Pad((w//2, h//2),padding_mode='reflect')]
        padding = Compose(aug_tmp)
        if w-10 < 0:
            down_side_w = 0
        else:
            down_side_w = -10
        if h-10 < 0:
            down_side_h = 0
        else:
            down_side_h = -10
        dh = random.randint(down_side_h, 10)
        dw = random.randint(down_side_w, 10)
        aug_tmp2 = [transforms.CenterCrop((h+dh,w+dw))]
        center_crop = Compose(aug_tmp2)
        if self.padding:
            img = padding(img)
        if self.imgnet:
            img = self.imgnetp(img)
        if self.hand_gen is not None:
            img = F.rotate(img, self.hand_gen.seed, PIL.Image.BILINEAR)
        else:
            img = self.rotate(img)
        if self.center_crop:
            img = center_crop(img)
        #print('img size: ',img.size)
        if not self.scale_resize:
            img = self.resize_crop(img)
        else:
            if random.random() < self.rc_proob:
                #img = self.resize_crop(img)
                try:
                    img = self.resize_crop(img)
                except:
                    img = self.resize(img)
            else:
                img = self.resize(img)
        if self.do_random_overlap:
            img = self.random_overlap(img)
        return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = _iterate_transforms(transform, x) 
        return x

def judge_area_ud(angle):
    if angle >= -90 and angle <= 90:
        return 0
    else:
        return 1