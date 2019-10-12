# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:30:16 2019

@author: Hsu
"""
'''
基于手写字符，采用卷积网络进行训练。对卷积层的输出进行可视化，对网络的倒数第二层进行
T-SNE可视化。
'''

from keras.datasets import mnist
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Input
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
#%%
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)
#%%
# design model
inp = Input((28, 28, 1), name='input')
conv_1 = Convolution2D(25, (5, 5),activation='relu',name='Conv_1')(inp)
pooling_1 = MaxPooling2D((2, 2), name='pooling_1')(conv_1)
conv_2 = Convolution2D(50, (5, 5),activation='relu',name='Conv_2')(pooling_1)
pooling_2 = MaxPooling2D((2, 2), name='pooling_2')(conv_2)
flatten = Flatten(name='flatten')(pooling_2)
dense_1 = Dense(50, activation='relu', name='dense_1')(flatten)

output = Dense(10, activation='softmax', name='output')(dense_1)

model = Model(inputs=inp, outputs=output)
model.summary()
model.layers
adam = Adam(lr=0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#%%
model.fit(X_train, y_train, batch_size=100, epochs=3)
#%%
from keras.utils import plot_model
plot_model(model, 'model.png', show_shapes=True)
#%%
inp = Input((28, 28, 1), name='input')
conv_1 = Convolution2D(25, (5, 5),activation='relu',name='Conv_1')(inp)
pooling_1 = MaxPooling2D((2, 2), name='pooling_1')(conv_1)
conv_2 = Convolution2D(50, (5, 5),activation='relu',name='Conv_2')(pooling_1)
pooling_2 = MaxPooling2D((2, 2), name='pooling_2')(conv_2)
flatten = Flatten(name='flatten')(pooling_2)
dense_1 = Dense(50, activation='relu', name='dense_1')(flatten)

model_2 = Model(inputs=inp, outputs=dense_1)

model_2.layers[1].set_weights(model.layers[1].get_weights())
model_2.layers[2].set_weights(model.layers[2].get_weights())
model_2.layers[3].set_weights(model.layers[3].get_weights())
model_2.layers[4].set_weights(model.layers[4].get_weights())
model_2.layers[5].set_weights(model.layers[5].get_weights())
model_2.layers[6].set_weights(model.layers[6].get_weights())

#%%
inp = Input((28, 28, 1), name='input')
conv_1 = Convolution2D(25, (5, 5),activation='relu',name='Conv_1')(inp)
model_3 = Model(inputs=inp, outputs=conv_1)

model_3.layers[1].set_weights(model.layers[1].get_weights())


#%%

hh = model.predict(X_test[:1000:])

hh = model_2.predict(X_test[:1000:])

hh = model_3.predict(X_test[:1000:])

plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    hhh  = hh[0][:,:,i]
    plt.imshow(hhh)

#%%  绘制T-sne图
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(hh)

b = y_test[:1000]
b = [list(i).index(1.0) for i in b]
plt.figure()
plt.scatter(x=result[:,0], y=result[:,1], c=b)

