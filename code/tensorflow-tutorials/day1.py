# -*- coding: utf-8 -*-#
'''
# Name:         day1
# Description:  
# Author:       super
# Date:         2020/9/8
'''

import tensorflow as tf

message = tf.constant('Hello world')
with tf.Session() as sess:
    print(sess.run(message).decode())

v1 = tf.constant([1,2,3,4])
v2 = tf.constant([5,6,7,8])
v_add = tf.add(v1, v2)
with tf.Session() as sess:
    print(sess.run(v_add))

def matrix_op():
    sess = tf.InteractiveSession()
    I_matrix = tf.eye(5)
    print(I_matrix.eval())
    print(I_matrix)

def matrix_op2():
    sess = tf.InteractiveSession()
    x = tf.Variable(tf.eye(10))
    
    x.initializer.run()
    print(x.eval())


if __name__ == '__main__':
    matrix_op2()