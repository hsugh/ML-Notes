# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:20:20 2019

@author: SHIJUNJUN
"""
'''
keras版本
'''

import numpy as np
import matplotlib.pyplot as plt
import keras

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) * x_data - 0.5 + noise

from keras.layers import Dense, Input
input_shape = Input(shape=(1,), name='input')
l1 = Dense(10, activation='relu')(input_shape)
prediction = Dense(1, name='output')(l1)

from keras.models import Model

model = Model(inputs=input_shape, outputs=prediction)
model.summary()

#model.compile(optimizer='SGD',
#              loss='mse',
#              metrics=['accuracy'])
model.compile(optimizer='SGD',
              loss='mse')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.show()

checkpoint = keras.callbacks.ModelCheckpoint(r'E:\PythonCode\TF'+'\weights.h5',
                                             monitor='loss',
                                             save_best_only=True,
                                             mode='min')

for i in range(2000):
    if i % 50 == 0:
        model.fit(x_data, y_data, epochs=50, callbacks=[checkpoint])
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
#        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        prediction_value = model.predict(x_data)
        lines = ax.plot(x_data, prediction_value, 'r', lw=5)
        plt.pause(0.1)
        
#%%
  
        
        

