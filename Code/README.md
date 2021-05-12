# ChestXray Convolutional Neural Network Ensemble 
## Introduction
---
Team 1335 Final Project for UIUC-CS598-Deep Learning in Healthcare on Topic A: Chest X-ray Disease Diagnosis
 
Our project implements an ChestXray diagnosis classification model by ensembling multiple Convolution Neural Network Model inspired by [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/).


## Dataset
---
Our Model is trained with [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) We split the whole dataset (112,120 images) into roughly 70% training data (78,468
images), 10% validation data (11,219 images), and 20% testing
data (22,433 images)Partitioned image folders, names and corresponding multi-hot vector [labels](./labels).

For example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0

## Prerequisites
---
- Python 3.4+
- [PyTorch](http://pytorch.org/) and its dependencies

## Usage
---
1. Clone this repository.

2. Run DataPreprocess.py to download and structure the data.

3. Run Main.py to Test the ensembled models

## Comparsion
---
We followed the training strategy described in the official paper, and a ten crop method is adopted both in validation and test. Compared with the original CheXNet, the per-class AUROC of our reproduced model is almost the same. We have also proposed a slightly-improved model which achieves a mean AUROC of 0.847 (v.s. 0.841 of the original CheXNet).

|     Pathology      | [Wang et al.](https://arxiv.org/abs/1705.02315) | [Yao et al.](https://arxiv.org/abs/1710.10501) | [CheXNet](https://arxiv.org/abs/1711.05225) | Our Ensemble Model | 
| :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: | 
|    Atelectasis     |                  0.716                   |                  0.772                   |                  0.8094                  |         0.83520          |      
|    Cardiomegaly    |                  0.807                   |                  0.904                   |                  0.9248                  |         0.91719          |      
|      Effusion      |                  0.784                   |                  0.859                   |                  0.8638                  |         0.89048          |      
|    Infiltration    |                  0.609                   |                  0.695                   |                  0.7345                  |         0.71766          |      
|        Mass        |                  0.706                   |                  0.792                   |                  0.8676                  |         0.87013          |       
|       Nodule       |                  0.671                   |                  0.717                   |                  0.7802                  |         0.80744          |       
|     Pneumonia      |                  0.633                   |                  0.713                   |                  0.7680                  |         0.78068          |       
|    Pneumothorax    |                  0.806                   |                  0.841                   |                  0.8887                  |         0.89177          |       
|   Consolidation    |                  0.708                   |                  0.788                   |                  0.7901                  |         0.81919          |       
|       Edema        |                  0.835                   |                  0.882                   |                  0.8878                  |         0.90103          |       
|     Emphysema      |                  0.815                   |                  0.829                   |                  0.9371                  |         0.93594          |       
|      Fibrosis      |                  0.769                   |                  0.767                   |                  0.8047                  |         0.85396          |       
| Pleural Thickening |                  0.708                   |                  0.765                   |                  0.8062                  |         0.79690          |      
|       Hernia       |                  0.767                   |                  0.914                   |                  0.9164                  |         0.93233          |       


## Contributors
---
Team Members: <br>
Difan Gu, Jiaqi Luo, Xialin Liu, Yangrui Fan <br>
{difangu2, jiaqi10, xialinl2, yangrui3}@illinois.edu <br>
Department of Computer Science,
University of Illinois at Urbana-Champaign, Urbana, IL 61801 <br>


## References:
---
Xinyu Weng, Nan Zhuang, Jingjing Tian and Yingcheng Liu., Machine Intelligence Lab, Institute of Computer Science & Technology, Peking University, Inplementation of CheXNet, (2018), GitHub repository, <br>
https://github.com/arnoweng/CheXNet

