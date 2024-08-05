## Load packages
print("\n Loading packages ...")

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, losses, optimizers, utils
from tensorflow.python.keras import backend as K

from pi_net import *

## Load images and ground-truth persistence images
print("\n Loading images and ground-truth PIs ...")
temp = sio.loadmat('Sample_Images_PI.mat')
imgs = temp['imgs']
PIs = temp['PIs']

## Load model and weights
print("\n Loading model and weights ...")
model = Image_PINet()
model.load_weights('PI-Net_CIFAR10.h5')

## Generate PIs using PI-Net
print("\n Generating PIs ...")
PIs_generated = model.predict(imgs)

## Saving generated PIs
if not os.path.exists('Examples'):
    os.makedirs('Examples')

j = 0
for i in range(len(imgs)):
    fig = plt.figure(figsize = (15,5))#,frameon=False)
    fig.add_subplot(131)
    plt.imshow(imgs[i])
    plt.title('Input Image',fontdict={'fontsize':20})

    fig.add_subplot(132)
    plt.imshow(PIs[i].reshape((3,50,50))[j])
    plt.colorbar()
    plt.clim(0,1)
    plt.title('Ground-truth PI',fontdict={'fontsize':20})

    fig.add_subplot(133)
    plt.imshow(PIs_generated[i].reshape((3,50,50))[j])
    plt.colorbar()
    plt.clim(0,0.8)
    plt.title('Generated PI',fontdict={'fontsize':20})
    
    fig.savefig('Examples/' + str(i+1) + '.png' )
    
print("\n Please go into 'Examples' folder to view saved images \n")
