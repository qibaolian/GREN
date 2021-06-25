# GREN
## GREN: Graph-Regularized Embedding Network for Weakly-Supervised Disease Localization in Chest X-rays

### Abstract

Locating diseases in chest X-ray images with few careful annotations is challenging. Recent works have tackled this problem with innovative weakly-supervised algorithms such as multi-instance learning (MIL) and class activation maps (CAM), however, these methods often yield inaccurate or incomplete regions. One of the reasons is that they overlooked the pathological implications hidden in the relationship across anatomical regions within each image and the relationship across images. In this paper, we argue that the cross-region and cross-image relationship, as compensating information, is vital to obtain more consistent and integral regions. To model the relationship, we propose the Graph Regularized Embedding Network (GREN), which leverages the intra-image and inter-image information to locate diseases on chest X-rays. GREN uses a pre-trained U-Net to segment the lung lobes, and then models the intra-image relationship between the lung lobes using an intra-image graph to observe different regions. Meanwhile, the relationship between in-batch images is modeled by an inter-image graph to compare multiple images. The above practice mimics the training and decision-making process of radiologists. In order for the deep embedding layers of the neural network to retain macro-structural information (important in the localization task), we use the Hash coding and Hamming distance to compute the graphs, which are used as regularizers to facilitate training. By means of this, our approach achieves the state-of-the-art result on NIH chest X-ray dataset for weakly-supervised disease localization. 

### Results
#![Image text](https://raw.githubusercontent.com/hongmaju/light7Local/master/img/productShow/20170518152848.png)

