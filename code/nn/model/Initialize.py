# -*- coding: utf-8 -*-#
'''
# Name:         Init
# Description:  
# Author:       super
# Date:         2020/11/25
'''

from enum import Enum

class NetType(Enum):
    '''
    标记网络类型
    '''
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

class InitialMethod(Enum):
    '''
    标记权重矩阵的初始化方法
    '''
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3