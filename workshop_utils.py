import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
import json

def extract_coordinates_for_all_frames(person_id,number_of_frames,names,points):
    return_dict = {}
    for name in names:
        return_dict[name] = []
    for i in range(number_of_frames):
        file_path=".\output_json\clip_00000000000"+str(i)+"_keypoints.json"
        f = open(file_path)
        data = json.load(f)
        for i,name in enumerate(names):
            return_dict[name].append(data['people'][person_id]['face_keypoints_2d'][points[i]])

    return return_dict
    # returns JSON object as
# a dictionary


def get_angle_between_three_points(point1, point2, point3):
    """
    :param point1: [x,y]
    :param point2: [x,y]
    :param point3: [x,y]
    :return: Angle between vector1 and vector2 in RADIANS
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle



