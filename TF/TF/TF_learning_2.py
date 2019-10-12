# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:11:36 2019

@author: SHIJUNJUN
"""
'''
Session的两种打开模式
'''
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                      [2]])
product = tf.matmul(matrix1, matrix2)  # 矩阵相乘

# 方法1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# 方法2
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
