# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:01:52 2018

@author: Nezam Avaran
"""
from keras.layers import *
from keras.models import Model,Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt



def l2(y_true, y_pred):
    return K.l2_normalize(y_pred-y_true)

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
    
    
    
def TrainGenerator(batch_size):
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
        
def ValidationGenerator(batch_size):
    while True:
        index=np.random.randint(low=1,high=200,size=batch_size)
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
  
#boxsize=100;
#input_shape = ( boxsize, boxsize,1)
#inputs = Input(input_shape)
#conv1=Conv2D(100, (3, 3),padding='same', activation='relu',kernel_initializer = 'he_normal')(inputs)
#conv1=Conv2D(100, (3, 3),padding='same', activation='relu',kernel_initializer = 'he_normal')(conv1)
#temp_reshape1=Reshape((100, 100, 100, 1))(conv1)
#pool1=MaxPooling2D(pool_size=(2, 2))(conv1)
#conv2=Conv2D(50, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(pool1)
#conv2=Conv2D(50, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(conv2)
#temp_reshape2=Reshape((50, 50, 50, 1))(conv2)
#pool2=MaxPooling2D(pool_size = (2, 2))(conv2)
#conv3=Conv2D(25, (3, 3), padding='same',kernel_initializer = 'he_normal',activation='relu')(pool2)
#conv3=Conv2D(25, (3, 3), padding='same', kernel_initializer = 'he_normal',activation='relu')(conv3)
##temp_reshape3=Reshape((25, 25, 25, 1))(conv3)
##pool3=MaxPooling2D(pool_size = (2, 2))(conv3)
##conv4=Conv2D(12, (3, 3), padding='same',kernel_initializer = 'he_normal',activation='relu')(pool3)
##conv4=Conv2D(12, (3, 3), padding='same', kernel_initializer = 'he_normal',activation='relu')(conv4)
#Drop1=Dropout(0.2)(conv3)
#reshape1=Reshape((25, 25, 25, 1))(Drop1)
##conv5=Conv3D(12, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(reshape1)
##conv5=Conv3D(12, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(conv5)
##up1=UpSampling3D(size = (2, 2, 2))(conv5)
##pad1=ZeroPadding3D(padding=((1,0), (1,0), (1,0)), data_format=None)(up1)
##merge1 = concatenate([temp_reshape3,pad1],axis=-1)
#conv6=Conv3D(50, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(reshape1)
#conv6=Conv3D(50, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(conv6)
#up2=UpSampling3D(size = (2, 2, 2))(conv6)
#merge2 = concatenate([temp_reshape2,up2],axis=-1)
#conv7=Conv3D(20, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(merge2)
#conv7=Conv3D(20, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(conv7)
#up3=UpSampling3D(size = (2, 2, 2))(conv7)
#merge2 = concatenate([temp_reshape1,up3],axis=-1)
#conv8=Conv3D(20, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(merge2)
#conv8=Conv3D(20, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(conv8)
#conv8=Conv3D(2, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu')(conv8)
#conv8=Conv3D(1, (3, 3, 3), padding='same',kernel_initializer = 'he_normal',activation='sigmoid')(conv8)
#
#model = Model(inputs,conv8)
#model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
#model.summary()
##
##    
##    
##    #cloud_path='depth_to_3d/data_3d.mat'
##    #depthMap_path='depth_to_3d/data_depth.mat'
##    #data_depth,data_3d=LoadData(cloud_path,depthMap_path)
##    #data_3d_pad=np.pad(data_3d,((0,0),(891,891),(0,0)),'constant', constant_values=(0))
##    #data_depth=data_depth.reshape(1000,300,300,1)
##    #data_3d_pad=data_3d_pad.reshape(1000,55272,3,1)
#fpath = "3d_v2_W/weights-3dv4_Shortcut-{epoch:02d}-{loss:.3f}.hdf5"
#callbacks = [ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
#             EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=1, mode='min')]
##
##    #history = model.fit(depth[:700].reshape(700,300,300,1), voxels[:700].reshape(700,300,300,300,1),
##    #                epochs=50,
##    #                batch_size=4,
##    #                shuffle=True,
##    #                validation_data=(depth[700:].reshape(300,300,300,1), voxels[700:].reshape(300,300,300,300,1)),
##    #                callbacks=callbacks)
##    
#history =model.fit_generator(TrainGenerator(5), samples_per_epoch =200, nb_epoch = 20, validation_steps=40,validation_data=ValidationGenerator(5), class_weight=None)
#model.save('3dv2_Final.h5')
#depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth1.mat')['I']
#model=load_model('3dv2_2_200ep.h5')
#pred=model.predict(depth.reshape(1,100,100,1))
faces_Test_depth=[]
l2_Test=[]
l2_tr_Test=[]
for i in range(200):
    print(i)
    depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth'+str(i+1)+'.mat')['I']
    #voxel=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\vox'+str(i+1)+'.mat')['OUTPUTgrid']
#    Pred=model.predict(depth.reshape(1,100,100,1))
#    Pred=Pred.reshape(100,100,100)
#    pred=np.copy(Pred)
#    np.place(pred,pred<0.1,0)
#    np.place(pred,pred>=0.1,1)
    #plt.imsave('v1_pred/testF/'+str(i+1)+'_Side_pred.png',np.rot90(np.rot90(np.sum(pred[:50,:,:],axis=0))))
    #plt.imsave('v1_pred/testF/'+str(i+1)+'_Side_voxel.png',np.rot90(np.rot90(np.sum(voxel[:50,:,:],axis=0))))
    #plt.imsave('v1_pred/testF/'+str(i+1)+'_Front_voxel.png',np.rot90(np.sum(np.moveaxis(voxel, -1, 0)[:,:,:],axis=0)))
    #plt.imsave('v1_pred/testF/'+str(i+1)+'_Front_pred.png',np.rot90(np.sum(np.moveaxis(pred, -1, 0)[:,:,:],axis=0)))
#    l2_Test.append(np.linalg.norm(np.subtract(voxel,Pred)))
#    l2_tr_Test.append(np.linalg.norm(np.subtract(voxel,pred)))
    faces_Test_depth.append(depth)
#l2_Train=[]
#l2_tr_Train=[]   
#for i in range(200,1000):
#    print(i)
#    depth=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\depth'+str(i+1)+'.mat')['I']
#    voxel=sio.loadmat('E:\\RS\\Kamyab\\depth_to_3d\\voxel_data100\\vox'+str(i+1)+'.mat')['OUTPUTgrid']
#    Pred=model.predict(depth.reshape(1,100,100,1))
#    Pred=Pred.reshape(100,100,100)
#    pred=np.copy(Pred)
#    np.place(pred,pred<0.1,0)
#    np.place(pred,pred>=0.1,1)
#    plt.imsave('v1_pred/deep-shortcut/train/'+str(i+1)+'_pred.png',np.sum(pred[:50,:,:],axis=0).T)
#    plt.imsave('v1_pred/deep-shortcut/train/'+str(i+1)+'_voxel.png',np.sum(voxel[:50,:,:],axis=0).T)
#    l2_Train.append(np.linalg.norm(np.subtract(voxel,Pred)))
#    l2_tr_Train.append(np.linalg.norm(np.subtract(voxel,pred)))