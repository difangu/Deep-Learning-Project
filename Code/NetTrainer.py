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

class ChexnetTrainer():
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        
        # NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'DENSE-NET-161': model = DenseNet161(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'VGG': model = VGG16(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'ALEX-NET': model = AlexNet(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
                
        # DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        # DATASET BUILDERS
        datasetTrain = ChestXrayDataSet(pathDirData, pathFileTrain, transformSequence, 0.7)
        datasetVal =   ChestXrayDataSet(pathDirData, pathFileVal, transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=2, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)
        
        # OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        # LOSS
        loss = torch.nn.BCELoss(reduction='mean')
        
        # checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print("loading from check point")

        # INITIAL LOSS
        
        lossMIN = 100000

        # TRAIN THE NETWORK
        
        for epochID in range (0, trMaxEpoch):
            print(epochID)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%m%d%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%m%d%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            
            print(lossVal)
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, '/gdrive/My Drive/CS598-Project-Data/' + 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda()
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        
        for i, (input, target) in enumerate (dataLoader):
            with torch.no_grad():
                target = target.cuda()

                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)    
                varOutput = model(varInput)
                
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                
                lossVal += losstensor.item()
                lossValNorm += 1
            
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
               
    #--------------------------------------------------------------------------------     
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC

    #--------------------------------------------------------------------------------  
    
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        
        # NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'DENSE-NET-161': model = DenseNet161(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'VGG': model = VGG16(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'ALEX-NET': model = AlexNet(nnClassCount, nnIsTrained).cuda()

        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        # DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = ChestXrayDataSet(pathDirData, pathFileTest, transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=2, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (input, target) in enumerate(dataLoaderTest):
            with torch.no_grad():
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = input.size()
                
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
                
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                
                outPRED = torch.cat((outPRED, outMean.data), 0)

        print(outGT)
        print(outPRED)
        print(list(outPRED.size()))
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
        return
