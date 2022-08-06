#Importing the necessary packages and libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import workshop_utils


if __name__ == '__main__':
    # get relevant dict for all frames
    names = ['mouth_upper_lip_bottom_1',
             'mouth_upper_lip_bottom_2',
             'mouth_upper_lip_bottom_3',
             'mouth_upper_lip_bottom_4',
             'mouth_upper_lip_bottom_5',
             'mouth_upper_lip_top_1',
             'mouth_upper_lip_top_2',
             'mouth_upper_lip_top_3',
             'mouth_lower_lip_bottom_1',
             'mouth_lower_lip_bottom_2',
             'mouth_lower_lip_bottom_3',
             'mouth_lower_lip_top_1',
             'mouth_lower_lip_top_2',
             'mouth_lower_lip_top_3',
             'mouth_left_edge',
             'mouth_right_edge']
    numbers = [60, 61, 62, 63, 64, 65, 66, 67, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 7, 8, 9, 31, 33, 35]