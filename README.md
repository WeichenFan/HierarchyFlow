# Hierarchy Flow For High-Fidelity Image-to-Image Translation

<div>
<div align="center">
    <a href='https://weichenfan.github.io/Weichen/' target='_blank'>Weichen Fan<sup>*,1</sup></a>&emsp;
    <a href='https://www.linkedin.com/in/jinghuan-chen/?originalSubdomain=sg' target='_blank'>Jinghuan Chen<sup>*,1</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://img.shields.io/badge/HierarchyFlow-v0.1-darkcyan)
![](https://img.shields.io/github/stars/WeichenFan/HierarchyFlow)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWeichenFan%2FHierarchyFlow&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://img.shields.io/badge/code%20style-black-000000.svg)

This repository contains the **official implementation** of the following paper:

> **Hierarchy Flow For High-Fidelity Image-to-Image Translation**<br>
> https://arxiv.org/abs/2308.06909
>
> **Abstract:** *Image-to-image (I2I) translation comprises a wide spectrum of tasks. Here we divide this problem into three levels: strong-fidelity translation, normal-fidelity translation, and weak-fidelity translation, indicating the extent to which the content of the original image is preserved. Although existing methods achieve good performance in weak-fidelity translation, they fail to fully preserve the content in both strong- and normal-fidelity tasks, e.g. sim2real, style transfer and low-level vision. In this work, we propose Hierarchy Flow, a novel flow-based model to achieve better content preservation during translation. Specifically, 1) we first unveil the drawbacks of standard flow-based models when applied to I2I translation. 2) Next, we propose a new design, namely hierarchical coupling for reversible feature transformation and multi-scale modeling, to constitute Hierarchy Flow. 3) Finally, we present a dedicated aligned-style loss for a better trade-off between content preservation and stylization during translation. Extensive experiments on a wide range of I2I translation benchmarks demonstrate that our approach achieves state-of-the-art performance, with convincing advantages in both strong- and normal-fidelity tasks.*

## Install
```shell
conda create -n lcl python=3.10
conda activate lcl
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
## Experiments
### 1. GTA2Cityscapes
Please download the [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/index.html) and [Cityscapes dataset (leftImg8bit)](https://www.cityscapes-dataset.com/downloads/) first.

Modify the configs/GTA2Cityscapes/config.yaml
```yaml
#change the source_root and target_root for train and test respectively
  train:
    source_list: 'datasets/GTA/train.txt'
    target_list: 'datasets/Cityscapes/train.txt'
    source_root: '{YOUR PATH TO GTA}'
    target_root: '{YOUR PATH TO CITYSCAPES}'

  test:
    source_list: 'datasets/GTA/test.txt'
    target_list: 'datasets/Cityscapes/test.txt'
    source_root: '{YOUR PATH TO GTA}'
    target_root: '{YOUR PATH TO CITYSCAPES}'
```
**Training**
```Shell
#FOR SLURM
bash scripts/GTA2CITY/train.sh partition GPU_NUM

#NO SLURM
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/config.yaml
```

**Test**
```Shell
#FOR SLURM
bash scripts/GTA2CITY/eval.sh partition GPU_NUM {ckpt_path}

#NO SLURM
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/config.yaml --load_path {ckpt_path}
```

## Todo

1. [x] Release the Code.
2. [x] Add GTA2Cityscapes exp.
3. [ ] Add COCO2Wikiart exp.
4. [ ] Add more exp.

## Cite
```bibtex
@article{fan2023hierarchy,
  title={Hierarchy Flow For High-Fidelity Image-to-Image Translation},
  author={Fan, Weichen and Chen, Jinghuan and Liu, Ziwei},
  journal={arXiv preprint arXiv:2308.06909},
  year={2023}
}
```


**Under construction...**

