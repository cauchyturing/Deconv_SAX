# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:27:54 2016

@author: Stephen-Lu
"""
import os 
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.local import LocallyConnected2D
from keras.models import Model
from utils import tile_raster_images
import keras
from sklearn import svm
from sklearn import cross_validation

def load(dir, rescale = False):
    direc = dir+'/train/'
    files = os.listdir(direc)
    channel_num_train = len(files)
    datum_train = []
    for f in files:
        data = open(direc + f, 'r').readlines()
        data = [np.asarray(map(float, each.strip().split('\t'))) for each in data]
        if f == files[0]:
            label_train = [each[0] for each in data]
            l = [each.size for each in data]
        else:
            assert label_train == [each[0] for each in data]
            l == [each.size for each in data]
        data = [(each[1:]-np.mean(each[1:]))/(np.std(each[1:])+0.001) for each in data]
        if rescale:
            data = [(each[1:]-np.min(each[1:]))/(np.max(each[1:]) - np.min(each[1:] + +0.001)) for each in data]
        datum_train.append(data)
#    del data
    #Reshape the data
    sample_num_train = len(datum_train[0])
    length_train = max(l)
    
    direc = dir+'/test/'
    files = os.listdir(direc)
    channel_num_test = len(files)
    datum_test = []
    for f in files:
        data = open(direc + f, 'r').readlines()
        data = [np.asarray(map(float, each.strip().split('\t'))) for each in data]
        if f == files[0]:
            label_test = [each[0] for each in data]
            l = [each.size for each in data]
        else:
            assert label_test == [each[0] for each in data]
            l == [each.size for each in data]
        data = [(each[1:]-np.mean(each[1:]))/(np.std(each[1:])+0.001) for each in data]
        if rescale:
            data = [(each[1:]-np.min(each[1:]))/(np.max(each[1:]) - np.min(each[1:] + +0.001)) for each in data]
        datum_test.append(data)
#    del data
    #Reshape the data
    sample_num_test = len(datum_test[0])
    length_test = max(l)
    
    assert channel_num_test == channel_num_train
    channel_num = channel_num_test
    
    length = max(length_train, length_test)
    
    def form_data(datum, sample_num, label):
        mat = []
        for s in xrange(sample_num):
            vec = np.asarray([])
            for c in xrange(channel_num):
                vec = np.hstack((vec,np.hstack((datum[c][s],np.full(length - datum[c][s].size,datum[c][s].min())))))
            mat.append(vec)
        mat = np.asarray(mat, dtype = np.float32)
        label = np.asarray(label, dtype = np.float32)
        return mat, label
    x_train, y_train = form_data(datum_train, sample_num_train, label_train)
    x_test, y_test = form_data(datum_test, sample_num_test, label_test)
    return (channel_num, length), (x_train, y_train), (x_test, y_test)
    
names = ['AUSLAN_MTS', 'JP_MTS','Libra_MTS','LP_MTS','MOCAP_MTS', 'wafer_MTS']
names = ['wafer_MTS']
for name in names:    
    #Read all data in memory and do normalization
    direc = 'MTSdata/'+ name 
    shape, (x_train, y_train), (x_test, y_test) = load(direc, True)  
     
    
#Convolutional AutoEncoder
input_img = Input(shape=(1, shape[0], shape[1]))

#x = Convolution2D(16, 1, 1, activation='relu', border_mode='same')(input_img)
#x = MaxPooling2D((2, 1), border_mode='same')(x)
#x = Convolution2D(8, 1, 1, activation='relu', border_mode='same')(x)
#x = MaxPooling2D((2, 1), border_mode='same')(x)
#x = Convolution2D(8, 1, 1, activation='relu', border_mode='same')(x)
#encoded = MaxPooling2D((2, 1), border_mode='same')(x)
#
## at this point the representation is (8, 4, 4) i.e. 128-dimensional
#
#x = Convolution2D(8, 1, 1, activation='relu', border_mode='same')(encoded)
#x = UpSampling2D((2, 1))(x)
#x = Convolution2D(8, 1, 1, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 1))(x)
#x = Convolution2D(16, 1, 1, activation='relu')(x)
#x = UpSampling2D((2, 1))(x)
#decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

#x = Convolution2D(16, shape[0], 6, activation='relu', border_mode='same')(input_img)
#encoded = MaxPooling2D((shape[0], 1), border_mode='valid')(x)
#x = Convolution2D(16,shape[0], 6, activation='relu', border_mode='same')(encoded)
#x = UpSampling2D((shape[0], 1))(x)
#decoded = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same')(x)

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((shape[0]/2, 1), border_mode='valid')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)
encoded = MaxPooling2D((shape[0]/2, 1), border_mode='valid')(x)
x = Convolution2D(8,3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((shape[0]/2, 1))(x)
x = Convolution2D(16,3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((shape[0]/2, 1))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
optimizer = keras.optimizers.Adadelta(lr=.1, rho=0.95, epsilon=1e-08)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

x_train = np.reshape(x_train, (len(x_train), 1, shape[0], shape[1]))
x_test = np.reshape(x_test, (len(x_test), 1, shape[0], shape[1]))

autoencoder.fit(x_train, x_train,
                nb_epoch=200,
                batch_size=5,
                verbose=1,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)
#
encoder = Model(input = input_img, output = encoded)
encoded_train = encoder.predict(x_train)
encoded_test = encoder.predict(x_test)

#%%
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    #plt.imshow(x_test[i].reshape(shape[0], shape[1]), extent=[0,shape[1],0,100])
    plt.plot(np.arange(shape[1]),x_test[i].reshape(shape[0], shape[1])[0], np.arange(shape[1]), x_test[i].reshape(shape[0], shape[1])[1],
             np.arange(shape[1]),x_test[i].reshape(shape[0], shape[1])[2], np.arange(shape[1]), x_test[i].reshape(shape[0], shape[1])[3],
             np.arange(shape[1]),x_test[i].reshape(shape[0], shape[1])[4], np.arange(shape[1]), x_test[i].reshape(shape[0], shape[1])[5])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n +1)
    #plt.imshow(decoded_imgs[i].reshape(shape[0], shape[1]), extent=[0,shape[1],0,100])
    plt.plot(np.arange(shape[1]),decoded_imgs[i].reshape(shape[0], shape[1])[0], np.arange(shape[1]), decoded_imgs[i].reshape(shape[0], shape[1])[1],
             np.arange(shape[1]),decoded_imgs[i].reshape(shape[0], shape[1])[2], np.arange(shape[1]), decoded_imgs[i].reshape(shape[0], shape[1])[3],
             np.arange(shape[1]),decoded_imgs[i].reshape(shape[0], shape[1])[4], np.arange(shape[1]), decoded_imgs[i].reshape(shape[0], shape[1])[5])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
plt.show()
#%%
#n = 10
#plt.figure(figsize=(20, 8))
#for i in range(n):
#    ax = plt.subplot(1, n, i+1)
#    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()
#%%
feature_train = encoded_train.reshape(encoded_train.shape[0],-1)
feature_test = encoded_test.reshape(encoded_test.shape[0],-1)
np.savez('draw_feature_train_'+ name,x = feature_train, y=y_train)
np.savez('draw_feature_test_'+ name ,x=feature_test, y=y_test)

clf = svm.LinearSVC()
clf.fit(feature_train, y_train)
print np.mean(cross_validation.cross_val_score(clf, feature_train, y_train, cv=len(y_train)*2/3))
print clf.score(feature_train, y_train)
print clf.score(feature_test, y_test)
##%%
#ff = 'D:\Dropbox\CoZzu\Conv_SAX\MTSdata\ECG_MTS\ECG_TRAIN'
#dd = open(ff, 'r').readlines()
#sample = np.NaN
#ft0 = open('ECG_TRAIN0', 'w')
#ft1 = open('ECG_TRAIN1', 'w')
#for each in dd:
#    d = each.strip().split()
#    if d[0] != sample:
#        ft0.write('\n'+d[2]+'\t')
#        ft1.write('\n'+d[2]+'\t')
#        sample = d[0]
#    ft0.write(d[3]+'\t')
#    ft1.write(d[4]+'\t')
#ft0.close()
#ft1.close()
    
#%%
#n = 5
#plt.figure(figsize=(20, 4))
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i + 1)
#    #plt.imshow(x_test[i].reshape(shape[0], shape[1]), extent=[0,shape[1],0,100])
#    plt.plot(np.arange(199),dd[0].reshape(-1, 199)[i+3])
#    ax.get_xaxis().set_visible(False)
#    #ax.get_yaxis().set_visible(False)
#    ax = plt.subplot(2, n, i + n + 1)
#    #plt.imshow(x_test[i].reshape(shape[0], shape[1]), extent=[0,shape[1],0,100])
#    plt.plot(np.arange(153),ccc[0].reshape(-1, 153)[i+5])
#    ax.get_xaxis().set_visible(False)
#plt.show()    