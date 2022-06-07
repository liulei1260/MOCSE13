# -*- coding: utf-8 -*-
"""
==========================
**Author**: Qian Wang, qian.wang173@hotmail.com
"""


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import json
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from skimage import io, transform
import torch.nn.functional as F
import cv2
import skimage.measure
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import argparse
import pdb
plt.ion()   # interactive mode
from model import CANNet
from model_mcnn import MCNN
from model_cffnet import CFFNet
from model_csrnet import CSRNet
from model_sanet import SANet
from model_tednet import TEDNet
#from cannet import CANNet
from myVgg import headCount_vgg16_bn, headCount_vgg16, headCount_vgg16_part
from myResnet import headCount_resnet50, headCount_resnet101
from myDensenet import headCount_densenet121, headCount_densenet161, headCount_densenet169, headCount_densenet201
from myShufflenet import headCount_shufflenet_v2_x0_5, headCount_shufflenet_v2_x1_0
from generate_density_map import generate_multi_density_map,generate_density_map
from myInception_v3 import headCount_inceptionv3
#from generate_density_map_adaptive_sigma import generate_density_map_adaptive_sigma


parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

parser.add_argument('--dataset', type=str, default='A')

parser.add_argument('--optimizer', type=str, default='Adam')

parser.add_argument('--batchsize', type=int, default=8)

parser.add_argument('--ratio', type=int, default=1, help='input output size ratio')

parser.add_argument('--patchsize', type=int, default=128)

parser.add_argument('--nPatches', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--scale', type=float, default=100., help='factor for scaling the density map pixel values for easy training')

parser.add_argument('--modelType', type=str, default='headCount_vgg16_bn',
                    help='model types: headCount_vgg16, headCount_vgg16_bn,      headCount_vgg16_part')

parser.add_argument('--cl', action='store_true', help='use curriculum loss')
                            
        
args = parser.parse_args()
if 'resnet' in args.modelType:
    args.ratio = 8
if 'densenet' in args.modelType:
    args.ratio = 8

print(args)
IMG_EXTENSIONS = ['.JPG','.JPEG','.jpg', '.jpeg', '.PNG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions, args):
    images = []
    dir = os.path.expanduser(dir)
    """
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
    """
    d = os.path.join(dir)
    if args.dataset == 'u18':
        d = os.path.join(dir,'images_resized_to_2048')
    for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    image_path = os.path.join(root, fname)
                    head,tail = os.path.split(root)
                    #label_path = os.path.join(head,'ground_truth','GT_'+fname[:-4]+'.mat')
                    L_name=fname[:-4]
                    label_path = os.path.join(head,'ground_truth',L_name[15:]+'.txt')
                    if args.dataset == 'u18':
                        label_path = os.path.join(head,
                                        'ground_truth_mat_resized_to_2048',
                                                    fname[:-4]+'_ann.mat')
                    item = [image_path, label_path]
                    images.append(item)

    return images

class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train',extensions=IMG_EXTENSIONS):
        self.samples = make_dataset(data_dir,extensions, args)
        self.image_dir = data_dir
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):        
        img_file,label_file = self.samples[idx]
        image = cv2.imread(img_file)
        height, width, channel = image.shape
        ##annPoints = scipy.io.loadmat(label_file)        
        ff=open(label_file,'r')
        label = json.load(ff)# for shanghaitech
        xs = [point['x']  for point in label]   ###before
        ys = [point['y']  for point in label]   ###before
        xxxx = list(zip(xs, ys))
        annPoints = np.asarray(xxxx)
        positions = generate_density_map(shape=image.shape,points=annPoints,f_sz=15,sigma=4)
        shortEdge = np.minimum(height,width)
        #tmp = np.random.randint(0.2*shortEdge,0.5*shortEdge)
        tsize = [args.patchsize,args.patchsize]
        #tsize = positions.shape
        if self.phase=='train':
            targetSize = tsize
        else:
            targetSize = tsize
        height, width, channel = image.shape
        if height < tsize[0] or width < tsize[1]:
            image = cv2.resize(image,(np.maximum(tsize[0]+2,height),np.maximum(tsize[1]+2,width)))
            count = positions.sum()
            max_value = positions.max()
            # down density map
            positions = cv2.resize(positions, (np.maximum(tsize[0]+2,height),np.maximum(tsize[1]+2,width)))
            count2 = positions.sum()
            positions = np.minimum(positions*count/(count2+1e-8),max_value*10)
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        # transpose from h x w x channel to channel x h x w
        image = image.transpose(2,0,1)
        
        numPatches = args.nPatches
        if self.phase == 'train':
            patchSet, countSet = getRandomPatchesFromImage(image,positions,targetSize,numPatches)
            x = np.zeros((patchSet.shape[0],3,targetSize[0],targetSize[1]))
            if self.transform:
              for i in range(patchSet.shape[0]):
                #transpose to original:h x w x channel
                x[i,:,:,:] = self.transform(np.uint8(patchSet[i,:,:,:]).transpose(1,2,0))
            patchSet = x
        if self.phase == 'val' or self.phase == 'test':
            patchSet, countSet = getAllFromImage(image, positions)
            patchSet[0,:,:,:] = self.transform(np.uint8(patchSet[0,:,:,:]).transpose(1,2,0))
        return patchSet, countSet

def getRandomPatchesFromImage(image,positions,target_size,numPatches):
    # generate random cropped patches with pre-defined size, e.g., 224x224
    imageShape = image.shape
    if np.random.random()>0.5:
        for channel in range(3):
            image[channel,:,:] = np.fliplr(image[channel,:,:])
        positions = np.fliplr(positions)
    patchSet = np.zeros((numPatches,3,target_size[0],target_size[1]))
    # generate density map
    countSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    for i in range(numPatches):
        topLeftX = np.random.randint(imageShape[1]-target_size[0]+1)#x-height
        #if imageShape[2]-target_size[1]-1 < 1:
        #    pdb.set_trace()
        topLeftY = np.random.randint(imageShape[2]-target_size[1]+1)#y-width
        thisPatch = image[:,topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        # pdb.set_trace()
        patchSet[i,:,:,:] = thisPatch
        # density map
        position = positions[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        #position = skimage.measure.block_reduce(position,(2,2),np.sum)
        position = position.reshape((1, position.shape[0], position.shape[1]))
        countSet[i,:,:,:] = position
    return patchSet, countSet

def getAllPatchesFromImage(image,positions,target_size):
    # generate all patches from an image for prediction
    nchannel,height,width = image.shape
    nRow = np.int(height/target_size[1])
    nCol = np.int(width/target_size[0])
    target_size[1] = np.int(height/nRow)
    target_size[0] = np.int(width/nCol)
    patchSet = np.zeros((nRow*nCol,3,target_size[1],target_size[0]))
    for i in range(nRow):
      for j in range(nCol):
        # pdb.set_trace()
        patchSet[i*nCol+j,:,:,:] = image[:,i*target_size[1]:(i+1)*target_size[1], j*target_size[0]:(j+1)*target_size[0]]
    return patchSet#, countSet

def getAllFromImage(image,positions):
    nchannel, height, width = image.shape
    patchSet =np.zeros((1,3,height, width))
    patchSet[0,:,:,:] = image[:,:,:]
    countSet = positions.reshape((1,1,positions.shape[0], positions.shape[1]))
    return patchSet, countSet

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop((256)),
        #transforms.Resize((128,128)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop((256)),
        #transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

#data_dir = './data/shanghaitech/part_'+args.dataset+'_final/'
data_dir = './test_image/'
if args.dataset == 'u18':
    data_dir = './data/ucf2018/'
image_datasets = {x: ShanghaiTechDataset(data_dir+x+'_data', 
                        phase=x, 
                        transform=data_transforms[x])
                    for x in ['train','test']}
image_datasets['val'] = ShanghaiTechDataset(data_dir+'train_data',
                                            phase='val',
                                            transform=data_transforms['val'])
## split the data into train/validation/test subsets
indices = list(range(len(image_datasets['train'])))
split = np.int(len(image_datasets['train'])*0.2)
batch_size = args.batchsize

val_idx = np.random.choice(indices, size=split, replace=False)
train_idx = indices#list(set(indices)-set(val_idx))
test_idx = range(len(image_datasets['test']))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetSampler(test_idx)

num_workers=2
train_loader = torch.utils.data.DataLoader(dataset=image_datasets['train'],batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(dataset=image_datasets['val'],batch_size=1,sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=image_datasets['test'],batch_size=1,sampler=test_sampler, num_workers=num_workers)

dataset_sizes = {'train':len(train_idx),'val':len(val_idx),'test':len(image_datasets['test'])}
dataloaders = {'train':train_loader,'val':val_loader,'test':test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, args=args):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae_val = 1e6
    best_mae_by_val = 1e6
    best_mae_by_test = 1e6
    best_mse_by_val = 1e6
    best_mse_by_test = 1e6
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
       
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print('Current learning rate:{}'.format(current_lr))
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Iterate over data.
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                labels = labels*args.scale
                labels3 = skimage.measure.block_reduce(labels.cpu().numpy(), 
                                                       (1,1,1,args.ratio,args.ratio),
                                                       np.sum)
                labels3 = torch.from_numpy(labels3)
                labels3 = labels3.to(device)
                inputs = inputs.to(device)
                inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
                labels3 = labels3.view(-1,labels3.shape[3],labels3.shape[4])
                inputs = inputs.float()
                labels3 = labels3.float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output3 = model(inputs)
                    loss3 = criterion(output3, labels3)
                    #th = 10*(epoch//100+1)
                    #th = 0.01*epoch+1 # cl4 for sanet (initial setting)
                    #th = 0.005*epoch+0.3 # cl5 for sanet (output size ratio 1)
                    #th = 0.01*epoch+5 # cl3 for mcnn
                    #th = 0.1*epoch+5 # cl2 for inceptionv3
                    #th = 0.1*epoch + 20 # for cannet
                    if args.cl:
                    	th = (0.001 * epoch + 0.05) * args.scale * args.ratio * args.ratio / 16.
                    else:
                    	th = 1e5 # no cl
                    weights = th/(F.relu(labels3-th)+th)
                    loss3 = loss3*weights
                    loss3 = loss3.sum()/weights.sum()
                    loss = loss3
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print()
        if epoch%1==0:
            tmp,epoch_mae,epoch_mse,epoch_mre=test_model_without_density(model,optimizer,'val')
            tmp,epoch_mae_test,epoch_mse_test,epoch_mre_test = test_model_without_density(model,optimizer,'test')
            if  epoch_mae < best_mae_val:
                best_mae_val = epoch_mae
                best_mae_by_val = epoch_mae_test
                best_mse_by_val = epoch_mse_test
                best_epoch_val = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch_mae_test < best_mae_by_test:
                best_mae_by_test = epoch_mae_test
                best_mse_by_test = epoch_mse_test
                best_epoch_test = epoch
            print()
            print('best MAE and MSE by val: {} and {} at Epoch {}'.format(best_mae_by_val,best_mse_by_val, best_epoch_val))
            print('best MAE and MSE by test: {} and {} at Epoch {}'.format(best_mae_by_test,best_mse_by_test, best_epoch_test))
        #if current_lr<1e-6:
        #    break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model,optimizer,phase):
    since = time.time()
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    densityArray=[]
    # Iterate over data.
    for index, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        labels = labels.view(-1,labels.shape[3],labels.shape[4])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs3 = model(inputs)
            outputs3 = outputs3.to(torch.device("cpu")).numpy()/args.scale
            densityArray.append(outputs3)
            pred_count = outputs3.sum()
        true_count = labels.to(torch.device("cpu")).numpy().sum()
        # backward + optimize only if in training phase
        mse = mse + np.square(pred_count-true_count)
        mae = mae + np.abs(pred_count-true_count)
        mre = mre + np.abs(pred_count-true_count)/true_count
        pred[index,0] = pred_count
        pred[index,1] = true_count
    pred = pred[0:index+1,:]
    mse = np.sqrt(mse/(index+1))
    mae = mae/(index+1)
    mre = mre/(index+1)
    print(phase+':')
    print(mae,mse,mre)
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #scipy.io.savemat('./results.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
    return pred,mae,mse,mre,densityArray
    # load best model weights
    # return cmap,emap,p,r,f,outputs_test.to(torch.device("cpu")).numpy()

def test_model_without_density(model,optimizer,phase):
    since = time.time()
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    # Iterate over data.
    for index, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        labels = labels.view(-1,labels.shape[3],labels.shape[4])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs3 = model(inputs)
            outputs3 = outputs3.to(torch.device("cpu")).numpy()/args.scale
            pred_count = outputs3.sum()
        true_count = labels.to(torch.device("cpu")).numpy().sum()
        # backward + optimize only if in training phase
        mse = mse + np.square(pred_count-true_count)
        mae = mae + np.abs(pred_count-true_count)
        mre = mre + np.abs(pred_count-true_count)/true_count
        pred[index,0] = pred_count
        pred[index,1] = true_count
    pred = pred[0:index+1,:]
    mse = np.sqrt(mse/(index+1))
    mae = mae/(index+1)
    mre = mre/(index+1)
    print(phase+':')
    print(mae,mse,mre)
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #scipy.io.savemat('./results.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
    return pred,mae,mse,mre

######################################################################
if args.modelType == 'headCount_vgg16':
    model = headCount_vgg16(pretrained=True)
elif args.modelType == 'headCount_vgg16_part':
    model = headCount_vgg16_part(pretrained=True)
elif args.modelType == 'headCount_vgg16_bn':
    model = headCount_vgg16_bn(pretrained=True)
elif args.modelType == 'headCount_resnet50':
    model = headCount_resnet50(pretrained=True)
elif args.modelType == 'headCount_resnet101':
    model = headCount_resnet101(pretrained=True)
elif args.modelType == 'headCount_densenet121':
    model = headCount_densenet121(pretrained=True)
elif args.modelType == 'headCount_densenet161':
    model = headCount_densenet161(pretrained=True)
elif args.modelType == 'headCount_densenet169':
    model = headCount_densenet169(pretrained=True)
elif args.modelType == 'headCount_densenet201':
    model = headCount_densenet201(pretrained=True)
elif args.modelType == 'headCount_shufflenet_v2_x0_5':
    model = headCount_shufflenet_v2_x0_5(pretrained=True)
elif args.modelType == 'headCount_shufflenet_v2_x1_0':
    model = headCount_shufflenet_v2_x1_0(pretrained=True)
elif args.modelType == 'headCount_inceptionv3':
    model = headCount_inceptionv3(pretrained=True)
elif args.modelType == 'TEDNet':
    model = TEDNet(use_bn=True)
else:
    print('Please select modelType from the list.')
#model = headCount_inceptionv3(pretrained=True)
#model = MCNN()
#model = SANet()
#model = CANNet()
# model = TEDNet(use_bn=True)
# model = models.inception_v3(pretrained=True)
# model.fc = nn.Linear(2048,1)
model = model.to(device)

criterion = nn.MSELoss(reduce=False)
#criterion = nn.L1Loss(reduce=False)

# Observe that all parameters are being optimized
# optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.95, weight_decay= 0)
#optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.95, weight_decay= 5e-4) # for MCNN
if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # inv3, vgg,
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr) # 
elif args.optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)# 
else:
    print('unknown optimizer...')
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=2000, verbose=True)

#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#
model_dir = './'
#model.load_state_dict(torch.load(model_dir+'sheepNet_model.pt'))
#test_model(model,optimizer,'test')
model = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=201, args=args)
pred,mae,mse,mre,densityArray = test_model(model,optimizer,'test')
scipy.io.savemat('./results_'+args.modelType+'_sh'+args.dataset+'_counting'+'.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre,'densityArray':densityArray})
torch.save(model.state_dict(), model_dir+'headCount_'+args.modelType+'_sh'+args.dataset+'_counting'+'.pt')

