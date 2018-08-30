# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:01:52 2018

@author: Nezam Avaran
"""
from keras.layers import Input, Dense, Conv3D, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization,MaxPooling3D,UpSampling3D,ZeroPadding3D,Dropout,Flatten,Reshape
from keras.models import Model,Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
    
    
    
def myGenerator(batch_size):
    while True:
        index=np.random.randint(low=200,high=1000,size=batch_size)
        depths=[]
        voxels=[]
        for i in index:
            depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth'+str(i)+'.mat')['I']
            voxel=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\vox'+str(i)+'.mat')['OUTPUTgrid']
            depths.append(depth)
            voxels.append(voxel)
           # print(i)
        depths=np.array(depths)
        voxels=np.array(voxels)
        depths=np.expand_dims(depths, axis=-1)
        voxels=np.expand_dims(voxels, axis=-1)
        #print('============>',depths.shape)
        #print('============>',voxels.shape)
        yield depths,voxels
  
model = Sequential()
boxsize=100;
model.add(Conv2D(10, (3, 3),
                        padding='same', 
                        activation='relu',
                        kernel_initializer = 'he_normal',
                        input_shape = ( boxsize, boxsize,1)))
model.add(Conv2D(10, (3, 3),
                        padding='same', 
                        activation='relu',
                        kernel_initializer = 'he_normal',
                        input_shape = ( boxsize, boxsize,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(20, (3, 3), 
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(Conv2D(20, (3, 3), 
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(25, (3, 3), 
                        padding='same', 
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(Conv2D(25, (3, 3), 
                        padding='same', 
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(Dropout(0.2))
model.add(Reshape((25, 25, 25, 1)))
model.add(Conv3D(50, (3, 3, 3),
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(Conv3D(50, (3, 3, 3),
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(UpSampling3D(size = (2, 2, 2)))
#model.add(Dropout(0.25))   
model.add(Conv3D(20, (3, 3, 3),
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(Conv3D(20, (3, 3, 3),
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(UpSampling3D(size = (2, 2, 2)))
model.add(Conv3D(2, (3, 3, 3),
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='relu'))
model.add(Conv3D(1, (3, 3, 3), 
                        padding='same',
                        kernel_initializer = 'he_normal',
                        activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
model.summary()
    
    
    #cloud_path='depth_to_3d/data_3d.mat'
    #depthMap_path='depth_to_3d/data_depth.mat'
    #data_depth,data_3d=LoadData(cloud_path,depthMap_path)
    #data_3d_pad=np.pad(data_3d,((0,0),(891,891),(0,0)),'constant', constant_values=(0))
    #data_depth=data_depth.reshape(1000,300,300,1)
    #data_3d_pad=data_3d_pad.reshape(1000,55272,3,1)
    
    #fpath = "weights-ae-{epoch:02d}-{loss:.3f}.hdf5"
    #callbacks = [ModelCheckpoint(fpath, monitor='loss', verbose=1, save_best_only=True, mode='min')]
    #history = model.fit(depth[:700].reshape(700,300,300,1), voxels[:700].reshape(700,300,300,300,1),
    #                epochs=50,
    #                batch_size=4,
    #                shuffle=True,
    #                validation_data=(depth[700:].reshape(300,300,300,1), voxels[700:].reshape(300,300,300,300,1)),
    #                callbacks=callbacks)
    
#history =model.fit_generator(myGenerator(20), samples_per_epoch =50, nb_epoch = 200, validation_data=None, class_weight=None)
#model.save('3dv2_2_200ep.h5')
#depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth1.mat')['I']
#model=load_model('3dv2_2_200ep.h5')
#pred=model.predict(depth.reshape(1,100,100,1))
#l2_Test=[]
#l2_tr_Test=[]
#for i in range(200):
#    print(i)
#    depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth'+str(i+1)+'.mat')['I']
#    voxel=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\vox'+str(i+1)+'.mat')['OUTPUTgrid']
#    Pred=model.predict(depth.reshape(1,100,100,1))
#    Pred=Pred.reshape(100,100,100)
#    pred=np.copy(Pred)
#    np.place(pred,pred<0.15,0)
#    np.place(pred,pred>=0.15,1)
#    plt.imsave('v1_pred/test/'+str(i+1)+'_pred.png',np.sum(pred[:50,:,:],axis=0).T)
#    plt.imsave('v1_pred/test/'+str(i+1)+'_voxel.png',np.sum(voxel[:50,:,:],axis=0).T)
#    l2_Test.append(np.linalg.norm(np.subtract(voxel,Pred)))
#    l2_tr_Test.append(np.linalg.norm(np.subtract(voxel,pred)))
#
#l2_Train=[]
#l2_tr_Train=[]   
#for i in range(200,1000):
#    print(i)
#    depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth'+str(i+1)+'.mat')['I']
#    voxel=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\vox'+str(i+1)+'.mat')['OUTPUTgrid']
#    Pred=model.predict(depth.reshape(1,100,100,1))
#    Pred=Pred.reshape(100,100,100)
#    pred=np.copy(Pred)
#    np.place(pred,pred<0.15,0)
#    np.place(pred,pred>=0.15,1)
#    plt.imsave('v1_pred/train/'+str(i+1)+'_pred.png',np.sum(pred[:50,:,:],axis=0).T)
#    plt.imsave('v1_pred/train/'+str(i+1)+'_voxel.png',np.sum(voxel[:50,:,:],axis=0).T)
#    l2_Train.append(np.linalg.norm(np.subtract(voxel,Pred)))
#    l2_tr_Train.append(np.linalg.norm(np.subtract(voxel,pred)))