import torch
import os
import numpy as np
import time
import sys

import Net 
import NetTrainer
import NetEnsembler

def runTrain():
    
    DENSENET201 = 'DENSE-NET-201'
    DENSENET121 = 'DENSE-NET-169'
    DENSENET121 = 'DENSE-NET-161'
    RESNET50 = 'RES-NET-50'
    ALEXNET = 'ALEX-NET'
    VGG = 'VGG'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%m%d%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    # Path to the directory with images
    pathDirData = '.'
    
    # Paths to the files with training, validation and testing sets.
    # Each file should contains pairs [path to image, output vector]
    # Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './labels/new_train_list.txt'
    pathFileVal = './labels/new_val_list.txt'
    pathFileTest = './labels/new_test_list.txt'


    # type of the network, is it pre-trained 
    # number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 14
    
    # batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 100
    
    # size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = '/gdrive/My Drive/CS598-Project-Data/' + 'Dense201-new-m-05042021-220552.pth.tar'
    print ('Training NN architecture = ', nnArchitecture)
    print (pathModel)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    

def runTest():

    DENSENET201 = 'DENSE-NET-201'
    DENSENET121 = 'DENSE-NET-169'
    DENSENET121 = 'DENSE-NET-161'
    RESNET50 = 'RES-NET-50'
    ALEXNET = 'ALEX-NET'
    VGG = 'VGG'

    pathDirData = '.'
    pathFileTest = './labels/new_test_list.txt'
    nnArchitecture = ALEXNET
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = '/gdrive/My Drive/CS598-Project-Data/m-05062021-053218.pth.tar'
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)


def runEnsemble():

    pathDirData = "./"
    pathFileTest = '/gdrive/My Drive/CS598-Project-Data/labels/new_test_list.txt'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224
    timestampLaunch = ''
    
    Ensemble_Test(pathDirData, pathFileTest, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

    if __name__ == __main__:
        #runTrain()
        #runTest()
        runEnsemble()
