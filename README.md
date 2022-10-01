# MOCSE13
We create the dataset MOCSE13 by using Unity to generate synthetic images for 13 different objects (fruits and vegetables). We propose deep model architectures for multi-class object counting and object size estimation. Our proposed models with different backbones are evaluated on the synthetic dataset. The experimental results provide a benchmark for multi-class object counting and size estimation and the synthetic dataset can be served as a proper testbed for future studies.
The dataset MOCSE13 can be download from https://www.dropbox.com/sh/8nqyj6ycgr7jltl/AACfmgH7eU-OT4FDrrC9luyma?dl=0
We could create customized data (new type of object) based on the requirements

# Single/multi-class object counting and size estimation
A Pytorch implementation of deep model based approach for single/multi-class object counting and size estimation problems

# How to use
1. Modify the data path and parameter settings as needed
2. Use this command to train and test: python3 single(or multi)-counting(or size).py  
   i.e: python3 multi_channels_size_estination

# Reference
@article{LIU2022105449,
title = {A benchmark for multi-class object counting and size estimation using deep convolutional neural networks},
journal = {Engineering Applications of Artificial Intelligence},
volume = {116},
pages = {105449},
year = {2022},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2022.105449},
url = {https://www.sciencedirect.com/science/article/pii/S0952197622004390},
author = {Zixu Liu and Qian Wang and Fanlin Meng},
keywords = {Multi-class object counting, Crowd counting, Object size estimation, Convolutional neural networks, Synthetic dataset}
}

# Contact
zixuxilan@gmail.com  qian.wang173@hotmail.com
