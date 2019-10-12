# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:21:04 2019

@author: SHIJUNJUN
"""

import tensorflow as tf

state = tf.Variable(0, name='counter')
#print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()  # 如果定义了变量，必须用此语句进行初始化

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
sess.close()
