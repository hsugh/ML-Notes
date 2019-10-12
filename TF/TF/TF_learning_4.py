# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:31:57 2019

@author: SHIJUNJUN
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)


output = input1 * input2

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))