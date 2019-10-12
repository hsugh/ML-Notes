# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:34:30 2019

@author: SHIJUNJUN
"""

import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)
print([y, y_grad])
