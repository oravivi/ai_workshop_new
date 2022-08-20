#Importing the necessary packages and libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from workshop_utils import *


if __name__ == '__main__':
    # get relevant dict for all frames
    names = ['chin_bottom_left',
             'chin_bottom_middle',
             'chin_bottom_right',
             'mustache_left_edge',
             'mustache_middle',
             'mustache_right_edge',
             'outer_lip_left_corner'
             'outer_upper_lip_1',
             'outer_upper_lip_2',
             'outer_upper_lip_3',
             'outer_upper_lip_4',
             'outer_upper_lip_5',
             'outer_lip_right_corner',
             'outer_bottom_lip_1',
             'outer_bottom_lip_2',
             'outer_bottom_lip_3',
             'outer_bottom_lip_4',
             'outer_bottom_lip_5',
             'inner_lip_left_corner',
             'inner_upper_lip_1',
             'inner_upper_lip_2',
             'inner_upper_lip_3',
             'inner_lip_right_corner',
             'inner_bottom_lip_1',
             'inner_bottom_lip_2',
             'inner_bottom_lip_3'
             ]
    mouth_features = {"right inner corner" : [63, 64, 65],
                      "right outer corner" : [53, 54, 55],
                      "left inner corner" : [61, 60, 67],
                      "left outer corner" : [49, 48, 59]}

    # numbers = [60, 61, 62, 63, 64, 65, 66, 67, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 7, 8, 9, 31, 33, 35]
    numbers = [7, 8, 9, 31, 33, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]