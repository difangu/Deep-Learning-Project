import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics import roc_auc_score

import ChestXrayDataset
import Net

class ChestXNetEnsemble():

    def Ensemble(pathDirData, pathFileTest, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
            CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            cudnn.benchmark = True

            model_161_path = "/gdrive/MyDrive/CS598-Project-Data/70pct-161-train-m-05042021-042720.pth.tar"
            model_169_path = "/gdrive/MyDrive/CS598-Project-Data/70pct-Densenet169-m-05062021-044952.pth.tar"
            model_201_path = "/gdrive/MyDrive/CS598-Project-Data/70pct-Dense201-new-m-05042021-220552.pth.tar"
            model_vgg16_path = "/gdrive/MyDrive/CS598-Project-Data/70pct-vgg-16-m-05052021-082423.pth.tar"
            model_alexnet_path = "/gdrive/MyDrive/CS598-Project-Data/70pct-Alexnet-new-m-05062021-053218.pth.tar"
            model_resnet50_path = "/gdrive/MyDrive/CS598-Project-Data/70pct-resnet50-m-05052021-191847.pth.tar"
            # NETWORK ARCHITECTURE, MODEL LOAD
            if has_cuda:

                model161 = DenseNet161(nnClassCount, nnIsTrained).cuda()
                model161 = torch.nn.DataParallel(model161).cuda() 
                model169 = DenseNet169(nnClassCount, nnIsTrained).cuda()
                model169 = torch.nn.DataParallel(model169).cuda() 
                model201 = DenseNet201(nnClassCount, nnIsTrained).cuda()
                model201 = torch.nn.DataParallel(model201).cuda() 
                model_vgg16 = VGG16(nnClassCount, nnIsTrained).cuda()
                model_vgg16 = torch.nn.DataParallel(model_vgg16).cuda()           
                model_alexnet = AlexNet(nnClassCount, nnIsTrained).cuda()
                model_alexnet = torch.nn.DataParallel(model_alexnet).cuda()
                model_resnet50 = ResNet50(nnClassCount, nnIsTrained).cuda()
                model_resnet50 = torch.nn.DataParallel(model_resnet50).cuda()     

            else:
                model161 = DenseNet161(nnClassCount, nnIsTrained)
                model161 = torch.nn.DataParallel(model161)
                model169 = DenseNet169(nnClassCount, nnIsTrained)
                model169 = torch.nn.DataParallel(model169) 
                model201 = DenseNet201(nnClassCount, nnIsTrained)
                model201 = torch.nn.DataParallel(model201) 
                model_vgg16 = VGG16(nnClassCount, nnIsTrained)
                model_vgg16 = torch.nn.DataParallel(model_vgg16)          
                model_alexnet = AlexNet(nnClassCount, nnIsTrained)
                model_alexnet = torch.nn.DataParallel(model_alexnet)
                model_resnet50 = ResNet50(nnClassCount, nnIsTrained)
                model_resnet50 = torch.nn.DataParallel(model_resnet50)   

            if has_cuda:

                model161.load_state_dict(torch.load(model_161_path)['state_dict'])
                model169.load_state_dict(torch.load(model_169_path)['state_dict'])
                model201.load_state_dict(torch.load(model_201_path)['state_dict'])
                model_vgg16.load_state_dict(torch.load(model_vgg16_path)['state_dict'])
                model_alexnet.load_state_dict(torch.load(model_alexnet_path)['state_dict'])
                model_resnet50.load_state_dict(torch.load(model_resnet50_path)['state_dict'])

            else:

                model161.load_state_dict(torch.load(model_161_path,map_location=torch.device('cpu'))['state_dict'])
                model169.load_state_dict(torch.load(model_169_path,map_location=torch.device('cpu'))['state_dict'])
                model201.load_state_dict(torch.load(model_201_path,map_location=torch.device('cpu'))['state_dict'])
                model_vgg16.load_state_dict(torch.load(model_vgg16_path,map_location=torch.device('cpu'))['state_dict'])
                model_alexnet.load_state_dict(torch.load(model_alexnet_path,map_location=torch.device('cpu'))['state_dict'])
                model_resnet50.load_state_dict(torch.load(model_resnet50_path,map_location=torch.device('cpu'))['state_dict'])

            # DATA TRANSFORMS, TEN CROPS
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    
            transformList = []
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
            transformSequence=transforms.Compose(transformList)
            
            # DATASET BUILDERS

            datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
            dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=2, shuffle=False, pin_memory=True)
            if has_cuda:
                outGT = torch.FloatTensor().cuda()
                outPRED = torch.FloatTensor().cuda()
            else:
                outGT = torch.FloatTensor()
                outPRED = torch.FloatTensor()

            model161.eval()
            model169.eval()
            model201.eval()
            model_vgg16.eval()
            model_alexnet.eval()
            model_resnet50.eval()

            for i, (input, target) in enumerate(dataLoaderTest):
                print(i)
                with torch.no_grad():
                    if has_cuda:
                        target = target.cuda()
                    outGT = torch.cat((outGT, target), 0)
                    bs, n_crops, c, h, w = input.size()
                    varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

                    out_161 = model161(varInput)
                    out_169 = model169(varInput)
                    out_201 = model201(varInput)
                    out_vgg = model_vgg16(varInput)
                    out_alex = model_alexnet(varInput)
                    out_res = model_resnet50(varInput)

                    outMean161 = out_161.view(bs, n_crops, -1).mean(1)
                    outMean169 = out_169.view(bs, n_crops, -1).mean(1)
                    outMean201 = out_201.view(bs, n_crops, -1).mean(1)
                    outMeanVgg = out_vgg.view(bs, n_crops, -1).mean(1)
                    outMeanAlex = out_alex.view(bs, n_crops, -1).mean(1)
                    outMeanRes = out_res.view(bs, n_crops, -1).mean(1)
                    
                    num_row = (outMean161.shape[0])
                    temp = []
                    weights = {'Alex':1, 'Vgg':1, 'Res':1, 'D161':1, 'D201':1, 'D169':1}

                    for i in range(num_row):
                        t = []
                        t.append([i*weights['D161'] for i in outMean161[i].tolist()])
                        t.append([i*weights['D169'] for i in outMean169[i].tolist()])
                        t.append([i*weights['D201'] for i in outMean201[i].tolist()])
                        t.append([i*weights['Vgg'] for i in outMeanVgg[i].tolist()])
                        t.append([i*weights['Alex'] for i in outMeanAlex[i].tolist()])
                        t.append([i*weights['Res'] for i in outMeanRes[i].tolist()])
                        temp.append(t)

                    temp = torch.Tensor(temp).cuda()
                    outMean = torch.mean(temp, dim=1)
                    outPRED = torch.cat((outPRED, outMean.data), 0)

            aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
            aurocMean = np.array(aurocIndividual).mean()
            
            print ('AUROC mean ', aurocMean)
            
            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])
            
            return