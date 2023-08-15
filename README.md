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
![](https://img.shields.io/badge/code%20style-black-000000.svg)

Code would be released soon.

## train
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/config.yaml`

## test
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/config.yaml --eval_only`
